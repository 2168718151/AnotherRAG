'use strict';

const $ = id => document.getElementById(id);
const state = {context:null, documents:[], total:0, offset:0, sessions:[], active:null, view:'knowledge', busy:false, importing:false, pendingDelete:null, pendingArchive:null, toastTimer:null, listToken:0};
const pendingKey = 'another-rag.pending-turn';
let pendingTurn = null;
try { pendingTurn = JSON.parse(sessionStorage.getItem(pendingKey) || 'null'); } catch {}

class ApiError extends Error {
  constructor(message, status=0, code='') { super(message); this.status=status; this.code=code; }
}
async function api(path, options={}) {
  let response;
  try { response=await fetch(path,{cache:'no-store',...options}); }
  catch { throw new ApiError('连接中断，请检查本地服务后核对结果。'); }
  const body=await response.json().catch(()=>({}));
  if (!response.ok) {
    const detail=body.detail, code=typeof detail==='object'&&!Array.isArray(detail)?detail?.code:'';
    const known={REVISION_CONFLICT:'会话已更新，请重新打开后再提交。',REQUEST_IN_PROGRESS:'请求仍在处理中，请稍后核对。',SESSION_BUSY:'会话有未完成请求，请先核对上一条。',SESSION_UNCERTAIN:'上一条请求结果未知，请先处理该记录。',SESSION_ARCHIVED:'会话已归档，请新建会话。',SCOPE_MISMATCH:'会话与当前知识库不匹配。'};
    throw new ApiError(known[code]||(response.status===412?'资料已变化，请重新打开删除确认。':response.status===503?'服务尚未就绪，请检查知识库、模型与 PostgreSQL 配置。':typeof detail==='string'?detail:detail?.message||'请求失败（'+response.status+'），请刷新后重试。'),response.status,code);
  }
  return body;
}
const post = body => ({method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
function savePending(value) { pendingTurn=value; try { value?sessionStorage.setItem(pendingKey,JSON.stringify(value)):sessionStorage.removeItem(pendingKey); } catch {} }
function currentSession() { return state.sessions.find(s=>s.session_id===state.active); }
function button(label, className, handler) { const b=document.createElement('button'); b.type='button'; b.className=className; b.textContent=label; b.addEventListener('click',handler); return b; }
function announce(message) { const box=$('toast'); box.textContent=message; box.classList.add('show'); clearTimeout(state.toastTimer); state.toastTimer=setTimeout(()=>box.classList.remove('show'),5000); }
function showView(view,push=true) {
  state.view=view; ['knowledge','library','chat'].forEach(name=>{$(name).hidden=name!==view;});
  if(view==='chat') { $('navChat').setAttribute('aria-current','page'); $('navKnowledge').removeAttribute('aria-current'); }
  else { $('navKnowledge').setAttribute('aria-current','page'); $('navChat').removeAttribute('aria-current'); }
  document.title='Another RAG · '+({knowledge:'知识库广场',library:'知识库资料',chat:'AI 问答'})[view];
  closeHistory(); if(push) history.pushState({view},'','#'+view);
}
function closeHistory() { $('historyPanel').classList.remove('open'); $('historyScrim').hidden=true; $('historyScrim').classList.remove('show'); $('historyToggle').setAttribute('aria-expanded','false'); }
function setControls() {
  const session=currentSession(), unresolved=session?.turns?.some(t=>['queued','running','unknown'].includes(t.status));
  const recovering=pendingTurn?.session===state.active&&pendingTurn?.database===state.context?.database_id;
  const disabled=state.busy||!state.context?.knowledge_ready||!state.context?.conversations_ready||!!unresolved||!!recovering||!!session?.loading||!!session?.loadError;
  $('chatInput').disabled=disabled; $('composer').querySelector('button').disabled=disabled;
  $('composer').setAttribute('aria-busy',String(state.busy)); $('newChat').disabled=state.busy||!state.context?.knowledge_ready||!state.context?.conversations_ready;
  $('askFromLibrary').disabled=!state.context?.knowledge_ready||!state.context?.conversations_ready;
  $('chooseFiles').disabled=state.importing||!state.context?.knowledge_ready;
  $('fileInput').disabled=state.importing||!state.context?.knowledge_ready;
  $('uploadPanel').setAttribute('aria-disabled',String(!state.context?.knowledge_ready));
}
function renderLibraries() {
  const grid=$('kbGrid'); grid.replaceChildren(); $('kbCount').textContent=state.context?'1 座已连接':'连接中';
  if(!state.context) return;
  const card=document.createElement('div'); card.className='kb-card';
  const tile=button('','kb-tile',openLibrary); tile.setAttribute('aria-label','打开知识库：'+state.context.name);
  const name=document.createElement('span'); name.className='kb-tile-name'; name.textContent=state.context.name;
  const arrow=document.createElement('span'); arrow.className='card-arrow'; arrow.setAttribute('aria-hidden','true');
  arrow.innerHTML='<svg viewBox="0 0 24 24"><path d="M5 12h14M13 6l6 6-6 6"/></svg>';
  tile.append(name,arrow); card.append(tile); grid.append(card);
  $('kbPickerName').textContent=state.context.name; $('kbPickerTrigger').disabled=true;
  $('kbPickerTrigger').title='当前服务绑定此知识库；切换需重新启动服务';
  $('kbPickerTrigger').removeAttribute('aria-haspopup'); $('kbPickerTrigger').querySelector('.kb-picker-chevron').hidden=true;
  $('chatDocCount').textContent=state.total+' 份资料';
}
async function loadDocuments(reset=true) {
  if(!state.context) return;
  const token=++state.listToken, q=$('docSearch').value.trim(), offset=reset?0:state.offset;
  $('docList').setAttribute('aria-busy','true');
  try {
    const page=await api('/v1/documents?'+new URLSearchParams({q,offset,limit:50}));
    if(token!==state.listToken) return;
    state.documents=reset?page.items:[...state.documents,...page.items]; state.offset=offset+page.items.length;
    if(!q) state.total=page.total;
    $('libraryTitle').textContent=state.context.name; $('librarySubtitle').textContent='资料保存在当前知识库，可继续导入或开始提问。';
    $('docCountLabel').textContent=page.total+' 份 TXT 资料'+(q?'匹配':''); $('chatDocCount').textContent=state.total+' 份资料';
    renderDocuments(page.total);
  } catch(error) {
    if(token!==state.listToken) return;
    const list=$('docList'); list.replaceChildren(); const note=document.createElement('p'); note.className='doc-empty'; note.textContent=error.message;
    list.append(note,button('重新加载','doc-action',()=>loadDocuments(true)));
  } finally { if(token===state.listToken) $('docList').setAttribute('aria-busy','false'); }
}
function renderDocuments(total) {
  const list=$('docList'); list.replaceChildren();
  if(!state.documents.length) {
    const empty=document.createElement('p'); empty.className='doc-empty'; empty.textContent=$('docSearch').value?'没有找到匹配的资料，换个关键词试试。':'还没有资料，选择上方 TXT 文件开始导入。'; list.append(empty); return;
  }
  for(const item of state.documents) {
    const row=document.createElement('div'); row.className='doc-row';
    const icon=document.createElement('span'); icon.className='doc-icon'; icon.setAttribute('aria-hidden','true'); icon.innerHTML='<svg viewBox="0 0 24 24"><path d="M7 3h7l4 4v14H7a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2Z"/><path d="M14 3v5h5M9 13h6M9 17h6"/></svg>';
    const info=document.createElement('div'); info.className='doc-info';
    const name=document.createElement('div'); name.className='doc-name'; name.textContent=item.filename;
    const meta=document.createElement('div'); meta.className='doc-meta'; meta.textContent=item.node_count+' 个片段 · '+(item.status==='indexed'?'已入库':'待检查')+' · '+item.source;
    info.append(name,meta); const actions=document.createElement('div'); actions.className='doc-actions';
    const view=button('查看','doc-action',()=>openDocument(item.document_id)); view.setAttribute('aria-label','查看资料：'+item.filename);
    const remove=button('删除','doc-action danger',()=>requestDeleteDocument(item.document_id)); remove.setAttribute('aria-label','删除资料：'+item.filename); remove.disabled=state.importing||!state.context?.knowledge_ready;
    actions.append(view,remove); row.append(icon,info,actions); list.append(row);
  }
  if(state.offset<total) list.append(button('加载更多资料','doc-action load-more',event=>{event.currentTarget.disabled=true;loadDocuments(false);}));
}
function openLibrary() { showView('library'); $('docSearch').value=''; loadDocuments(true); }
async function openDocument(id) {
  $('documentDialogTitle').textContent='资料预览'; $('documentDialogMeta').textContent='正在读取…'; $('documentPreview').textContent=''; $('documentDialog').showModal();
  try {
    const item=await api('/v1/documents/'+encodeURIComponent(id)+'/preview');
    $('documentDialogTitle').textContent=item.filename; $('documentDialogMeta').textContent=state.context.name+' · '+item.total_characters+' 字符 · '+item.node_count+' 个片段';
    $('documentPreview').textContent=item.text+(item.truncated?'\n\n（仅展示前 8000 字符）':'');
  } catch(error) { $('documentPreview').textContent=error.message; $('documentDialogMeta').textContent='读取失败，可关闭后重试。'; }
}
async function requestDeleteDocument(id) {
  if(state.importing||!state.context?.knowledge_ready) return;
  try {
    const detail=await api('/v1/documents/'+encodeURIComponent(id)); state.pendingDelete=detail;
    $('deleteDocumentName').textContent='「'+detail.filename+'」'; $('deleteDocumentDialog').showModal();
  } catch(error) { announce(error.message); }
}
async function deleteDocument() {
  const detail=state.pendingDelete; if(!detail) return; const confirm=$('confirmDeleteDocument'); confirm.disabled=true;
  try {
    await api('/v1/documents/'+encodeURIComponent(detail.document_id),{method:'DELETE',headers:{'X-Database-ID':detail.database_id,'If-Match':detail.etag}});
    state.pendingDelete=null; $('deleteDocumentDialog').close(); announce('资料已从知识库删除，源 TXT 文件保留。'); await loadDocuments(true);
  } catch(error) { announce(error.message); } finally { confirm.disabled=false; }
}
function uploadStatus(message,error=false) { const box=$('uploadStatus'); box.hidden=false; box.textContent=message; box.classList.toggle('error',error); }
async function importFiles(fileList) {
  if(state.importing||!state.context?.knowledge_ready) return; const files=Array.from(fileList||[]); if(!files.length) return;
  state.importing=true; setControls(); $('uploadPanel').setAttribute('aria-busy','true'); const errors=[]; let done=0,duplicate=0;
  try {
    for(let i=0;i<files.length;i++) {
      const file=files[i]; uploadStatus('正在导入 '+(i+1)+'/'+files.length+'：'+file.name);
      if(!file.name.toLowerCase().endsWith('.txt')||file.size>50*1024*1024||!file.size) { errors.push(file.name+'：仅支持非空且不超过 50 MB 的 TXT'); continue; }
      try { const data=new FormData(); data.append('file',file); const result=await api('/v1/qa/ingest',{method:'POST',body:data}); done++; if(result.duplicate) duplicate++; }
      catch(error) { errors.push(file.name+'：'+error.message); if(!error.status||error.status===503) {errors.push('其余文件未继续导入，请先核对资料列表。');break;} }
    }
  } finally {
    state.importing=false; setControls(); $('fileInput').value=''; $('uploadPanel').setAttribute('aria-busy','false');
    uploadStatus('完成 '+done+' 份'+(duplicate?'（其中 '+duplicate+' 份已存在）':'')+(errors.length?'；'+errors.join('；'):''),errors.length>0);
    $('docSearch').value=''; await loadDocuments(true);
  }
}
async function loadSessions() {
  const rows=await api('/v1/conversations');
  const existing=new Map(state.sessions.map(session=>[session.session_id,session]));
  state.sessions=rows.map(row=>Object.assign(existing.get(row.session_id)||{},row));
  renderSessions(); return rows;
}
function renderSessions() {
  const root=$('sessionList'); root.replaceChildren(); const query=$('historySearch').value.trim().toLowerCase();
  const matches=state.sessions.filter(s=>s.name.toLowerCase().includes(query));
  if(!matches.length) {
    const p=document.createElement('p'); p.className='no-sessions'; p.textContent=state.context?.conversations_ready?(query?'没有匹配的会话':'还没有会话'):'会话服务未就绪'; root.append(p); return;
  }
  for(const session of matches) {
    const row=document.createElement('div'); row.className='session-row';
    const open=button('','session-item'+(session.session_id===state.active?' active':''),()=>openSession(session.session_id));
    open.setAttribute('aria-label','打开会话：'+session.name); const title=document.createElement('strong'); title.textContent=session.name;
    const meta=document.createElement('small'); meta.textContent=state.context.name+' · '+new Date(session.updated_at).toLocaleString('zh-CN',{month:'numeric',day:'numeric',hour:'2-digit',minute:'2-digit'});
    open.append(title,meta); open.disabled=state.busy;
    const archive=button('','session-delete',()=>requestArchive(session.session_id)); archive.setAttribute('aria-label','归档会话：'+session.name); archive.title='归档会话'; archive.disabled=state.busy;
    archive.innerHTML='<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 4h16v4H4zM6 8v12h12V8M10 12h4"/></svg>';
    row.append(open,archive); root.append(row);
  }
}
function turnMessages(turn) {
  const items=[{role:'user',text:turn.user_message??'旧记录未保存用户原话'}]; let status='answered',text=turn.assistant_answer;
  if(turn.status==='unknown') {status='unknown';text='请求结果未知，请核对后再继续。';}
  else if(['running','queued'].includes(turn.status)) {status='running';text='请求仍在处理中，请稍后核对结果。';}
  else if(turn.status==='failed') {status='error';text=turn.failure_code==='UNKNOWN_ACKNOWLEDGED'?'已确认结果未知，原请求未重新发送。':'本轮未完成（'+(turn.failure_code||'请求失败')+'）。';}
  else if(turn.answer_not_retained||text==null) {status='missing';text='旧记录未保存回答正文，无法还原。';}
  else if(text.trim()==='无') status='no_answer';
  items.push({role:'assistant',status,text,request_id:turn.request_id,turn_id:turn.turn_id}); return items;
}
function renderMessages() {
  const root=$('messages'); root.replaceChildren(); const session=currentSession(); $('chatTitle').textContent=session?.name||'新会话';
  if(session?.loading) {const p=document.createElement('p');p.textContent='正在读取历史对话…';p.setAttribute('role','status');root.append(p);setControls();return;}
  if(session?.loadError) {const p=document.createElement('p');p.textContent=session.loadError;root.append(p,button('重新加载会话','doc-action',()=>openSession(session.session_id)));setControls();return;}
  const items=session?.turns?.flatMap(turnMessages)||[];
  if(session?.optimistic) items.push({role:'user',text:session.optimistic.message},{role:'assistant',status:session.optimistic.error?'uncertain':'loading',text:session.optimistic.error||'正在检索资料并整理回答…'});
  if(!items.length) {
    const welcome=document.createElement('div');welcome.className='welcome';
    welcome.innerHTML='<div class="welcome-mark" aria-hidden="true"><svg viewBox="0 0 24 24"><path d="M12 3 14.5 9.5 21 12l-6.5 2.5L12 21l-2.5-6.5L3 12l6.5-2.5L12 3Z"/></svg></div><h2>让你的储备，回答你的问题。<span class="welcome-translation">Let your brain answer your questions.</span></h2>';
    root.append(welcome);
  }
  for(let i=0;i<items.length;i++) {
    const item=items[i], row=document.createElement('div'); row.className='message '+item.role+(['error','unknown','uncertain'].includes(item.status)?' error':item.status==='loading'?' loading':item.status==='no_answer'?' no-answer':'');
    const meta=document.createElement('div');meta.className='message-meta';meta.innerHTML=item.role==='user'?'<span class="user-avatar">你</span><span>你的问题</span>':'<span class="assistant-avatar" aria-hidden="true">✳</span><span>Another RAG</span>';
    const bubble=document.createElement('div');bubble.className='message-bubble';bubble.textContent=item.text;
    if(item.role==='assistant') {const tag=document.createElement('span');tag.className='status-tag';tag.textContent=({loading:'整理资料中',running:'处理中',unknown:'结果未知',uncertain:'待核对',error:'未完成',missing:'正文未保留',no_answer:'资料未覆盖'})[item.status]||'已回答';meta.append(tag);}
    if(item.status==='loading'){bubble.classList.add('loading-dots');bubble.setAttribute('role','status');}
    row.append(meta,bubble);
    if(item.status==='no_answer') row.append(button('前往资料页','answer-retry',openLibrary));
    if(item.status==='error'&&!state.busy) row.append(button('重新编辑问题','answer-retry',()=>{$('chatInput').value=items[i-1]?.text||'';$('chatInput').focus();}));
    if(['unknown','running','uncertain'].includes(item.status)) row.append(button('核对结果','answer-retry',()=>reconcile(session,true)));
    if(item.status==='unknown') row.append(button('确认结果未知并继续','answer-retry',()=>acknowledgeUnknown(session,item.request_id)));
    if(item.role==='assistant'&&item.turn_id&&item.status==='answered') row.append(button('查看回答依据','answer-retry',event=>showSources(session,item.turn_id,event.currentTarget)));
    root.append(row);
  }
  if(session?.nextCursor) root.append(button('继续加载对话','doc-action load-more',()=>loadMoreTurns(session)));
  if(session?.recoverable) root.append(button('使用原请求 ID 重发','answer-retry',()=>submitPayload(session,session.recoverable)));
  setControls(); $('conversationScroll').scrollTop=$('conversationScroll').scrollHeight;
}
async function openSession(id) {
  if(state.busy) return; const session=state.sessions.find(s=>s.session_id===id);if(!session)return;
  state.active=id;session.loading=true;session.loadError=null;session.optimistic=null;$('chatInput').value='';showView('chat');renderSessions();renderMessages();
  try {
    session.turns=await loadAllTurns(id);
    if(pendingTurn?.session===id&&pendingTurn?.database===state.context.database_id) await reconcile(session);
  } catch(error) {session.loadError=error.message;}
  finally {session.loading=false;renderMessages();}
}
async function loadAllTurns(id) {
  const turns=[];let cursor=null;
  do {
    const page=await api('/v1/conversations/'+id+'/turns?'+new URLSearchParams({limit:100,...(cursor?{cursor}:{})}));
    turns.push(...page.items);cursor=page.next_cursor;
  } while(cursor);
  return turns;
}
async function loadMoreTurns(session) {
  try {const page=await api('/v1/conversations/'+session.session_id+'/turns?'+new URLSearchParams({limit:100,cursor:session.nextCursor}));session.turns.push(...page.items);session.nextCursor=page.next_cursor;if(state.active===session.session_id)renderMessages();}
  catch(error){announce(error.message);}
}
function startNewChat() {if(state.busy)return;state.active=null;$('chatInput').value='';showView('chat');renderSessions();renderMessages();if(!$('chatInput').disabled)$('chatInput').focus();}
async function reconcile(session,checkOwner=false) {
  try {
    session.turns=await loadAllTurns(session.session_id);session.nextCursor=null;
    if(checkOwner) {
      for(const turn of session.turns.filter(t=>t.status==='running')) {
        try {await api('/v1/conversations/'+session.session_id+'/requests/'+encodeURIComponent(turn.request_id)+'/recover-orphan',post({}));}
        catch(error) {if(!['REQUEST_IN_PROGRESS','OWNER_UNVERIFIABLE'].includes(error.code))throw error;announce(error.message);}
      }
      session.turns=await loadAllTurns(session.session_id);
    }
    session.optimistic=null;session.recoverable=null;
    if(pendingTurn?.session===session.session_id&&pendingTurn?.database===state.context.database_id) {
      const found=session.turns.find(t=>t.request_id===pendingTurn.payload.request_id);
      if(found) savePending(null); else {session.optimistic={message:pendingTurn.payload.message,error:'未找到已认领记录。可使用原请求 ID 重发。'};session.recoverable=pendingTurn.payload;}
    }
    await loadSessions();if(state.active===session.session_id)renderMessages();
  } catch(error) {announce(error.message);}
}
async function submitPayload(session,payload) {
  if(state.busy)return;state.busy=true;session.recoverable=null;session.optimistic={message:payload.message};
  savePending({database:state.context.database_id,session:session.session_id,payload});renderSessions();renderMessages();
  try {
    const result=await api('/v1/conversations/'+session.session_id+'/turns',post(payload));session.revision=result.revision;
    if(result.status==='unknown')announce('本轮结果未知，请核对后处理。');await reconcile(session);
  } catch(error) {
    session.optimistic.error=error.message+' 请先核对结果，避免重复提交。';announce(error.message);await reconcile(session);
  } finally {state.busy=false;renderSessions();if(state.active===session.session_id)renderMessages();}
}
async function sendQuestion(event) {
  event.preventDefault();const input=$('chatInput'),message=input.value.trim();if(!message||input.disabled||state.busy)return;
  let session=currentSession();state.busy=true;setControls();renderSessions();
  try {
    if(!session) {
      const name=Array.from(message.replace(/[\x00-\x1f\x7f]/g,' ').trim()).slice(0,60).join('')||'新会话';
      session=await api('/v1/conversations',post({name}));session.turns=[];state.sessions.unshift(session);state.active=session.session_id;
    } else {await loadSessions();session=currentSession();if(!session)throw new ApiError('会话已归档，请新建会话。');}
    input.value='';
  } catch(error) {announce(error.message);return;}
  finally {state.busy=false;setControls();renderSessions();}
  await submitPayload(session,{request_id:crypto.randomUUID(),expected_revision:session.revision,message,use_history:true});
}
async function requestArchive(id) {
  if(state.busy)return;
  try {await loadSessions();const session=state.sessions.find(s=>s.session_id===id);if(!session)return;state.pendingArchive={id,revision:session.revision};$('deleteSessionName').textContent='「'+session.name+'」';$('deleteSessionDialog').showModal();}
  catch(error){announce(error.message);}
}
async function archiveSession() {
  const target=state.pendingArchive;if(!target)return;$('confirmDeleteSession').disabled=true;
  try {
    await api('/v1/conversations/'+target.id+'/archive',post({expected_revision:target.revision}));
    if(state.active===target.id)state.active=null;if(pendingTurn?.session===target.id)savePending(null);
    $('deleteSessionDialog').close();state.pendingArchive=null;await loadSessions();renderMessages();announce('会话已归档，记录仍保留。');
  } catch(error) {announce(error.message);} finally {$('confirmDeleteSession').disabled=false;}
}
async function acknowledgeUnknown(session,requestId) {
  if(!confirm('确认这条请求结果仍未知？确认后解除会话阻断，不会重新发送原问题。'))return;
  try {await api('/v1/conversations/'+session.session_id+'/requests/'+encodeURIComponent(requestId)+'/acknowledge-unknown',post({acknowledge_result_unknown:true}));await reconcile(session);}
  catch(error){announce(error.message);}
}
async function showSources(session,turnId,trigger) {
  trigger.disabled=true;
  try {
    const questions=await api('/v1/conversations/'+session.session_id+'/questions'),versions=[];
    for(const q of questions) versions.push(...await api('/v1/conversations/'+session.session_id+'/questions/'+q.question_id+'/versions'));
    const refs=versions.filter(v=>v.source_turn_id===turnId).flatMap(v=>v.evidence_refs||[]),ids=[...new Set(refs.map(r=>r.document_id))];
    const box=document.createElement('div');box.className='answer-sources';
    if(!ids.length){const p=document.createElement('p');p.className='answer-note';p.textContent='本轮没有可展示的资料引用。';box.append(p);}
    for(const id of ids) {
      try {const d=await api('/v1/documents/'+encodeURIComponent(id));box.append(button(d.filename,'source-link',()=>openDocument(id)));}
      catch {const p=document.createElement('p');p.className='answer-note';p.textContent='引用资料已删除或暂不可读取。';box.append(p);}
    }
    trigger.replaceWith(box);
  } catch(error) {trigger.disabled=false;announce(error.message);}
}
async function bootstrap() {
  const hint=$('serviceStatus'),connectionError=$('connectionError');connectionError.hidden=true;connectionError.replaceChildren();hint.textContent='正在连接本地服务…';state.context=null;setControls();
  try {
    state.context=await api('/v1/ui/context');renderLibraries();await loadDocuments(true);
    if(!state.context.knowledge_ready) hint.textContent='知识库尚未就绪，请检查服务状态。问答与资料操作已停用。';
    else if(state.context.conversations_ready){await loadSessions();hint.textContent='已连接当前知识库。建库、整库删除与切换请使用现有命令行。';}
    else hint.textContent='资料管理已连接；长期会话未就绪，请检查 PostgreSQL 配置及 schema 5 迁移。';
    renderSessions();renderMessages();
    if(pendingTurn?.database===state.context.database_id&&state.sessions.some(s=>s.session_id===pendingTurn.session))await openSession(pendingTurn.session);
  } catch(error) {hint.textContent=error.message;connectionError.hidden=false;connectionError.append(document.createTextNode(error.message+' '),button('重新连接','doc-action',bootstrap));renderSessions();renderMessages();}
  finally {setControls();}
}
function searchDocuments() {clearTimeout(searchTimer);searchTimer=setTimeout(()=>loadDocuments(true),250);}
let searchTimer;
$('navKnowledge').addEventListener('click',()=>showView('knowledge'));
$('navChat').addEventListener('click',()=>{showView('chat');renderMessages();});
$('backToPlaza').addEventListener('click',()=>showView('knowledge'));
$('askFromLibrary').addEventListener('click',startNewChat);
$('newChat').addEventListener('click',startNewChat);
$('chooseFiles').addEventListener('click',()=>$('fileInput').click());
$('fileInput').addEventListener('change',event=>importFiles(event.target.files));
$('docSearch').addEventListener('input',searchDocuments);
$('historySearch').addEventListener('input',renderSessions);
const upload=$('uploadPanel');
['dragenter','dragover'].forEach(type=>upload.addEventListener(type,event=>{event.preventDefault();if(!state.importing)upload.classList.add('drag-over');}));
upload.addEventListener('dragleave',event=>{if(!upload.contains(event.relatedTarget))upload.classList.remove('drag-over');});
upload.addEventListener('drop',event=>{event.preventDefault();upload.classList.remove('drag-over');importFiles(event.dataTransfer.files);});
['closeDocumentDialog','doneDocumentDialog'].forEach(id=>$(id).addEventListener('click',()=>$('documentDialog').close()));
$('cancelDeleteDocument').addEventListener('click',()=>{state.pendingDelete=null;$('deleteDocumentDialog').close();});
$('confirmDeleteDocument').addEventListener('click',deleteDocument);
$('cancelDeleteSession').addEventListener('click',()=>{state.pendingArchive=null;$('deleteSessionDialog').close();});
$('confirmDeleteSession').addEventListener('click',archiveSession);
$('historyToggle').addEventListener('click',()=>{$('historyPanel').classList.add('open');$('historyScrim').hidden=false;$('historyScrim').classList.add('show');$('historyToggle').setAttribute('aria-expanded','true');$('historySearch').focus();});
$('closeHistory').addEventListener('click',closeHistory);$('historyScrim').addEventListener('click',closeHistory);
$('chatInput').addEventListener('keydown',event=>{if(event.key==='Enter'&&!event.shiftKey&&!event.isComposing){event.preventDefault();$('composer').requestSubmit();}});
$('composer').addEventListener('submit',sendQuestion);
window.addEventListener('popstate',()=>{const view=location.hash.slice(1);showView(['knowledge','library','chat'].includes(view)?view:'knowledge',false);});
showView(['knowledge','library','chat'].includes(location.hash.slice(1))?location.hash.slice(1):'knowledge',false);
bootstrap();
