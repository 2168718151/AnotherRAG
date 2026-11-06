CREATE TABLE conversation_sessions (
    session_id UUID PRIMARY KEY,
    name TEXT NOT NULL CHECK (char_length(name) BETWEEN 1 AND 64),
    database_id UUID NOT NULL,
    storage_fingerprint CHAR(64) NOT NULL,
    revision BIGINT NOT NULL DEFAULT 0 CHECK (revision >= 0),
    archived_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX conversation_sessions_scope_idx
    ON conversation_sessions (database_id, storage_fingerprint, updated_at DESC);

CREATE TABLE turn_requests (
    session_id UUID NOT NULL REFERENCES conversation_sessions(session_id) ON DELETE CASCADE,
    turn_id UUID NOT NULL,
    request_id TEXT NOT NULL CHECK (char_length(request_id) BETWEEN 1 AND 128),
    request_digest CHAR(64) NOT NULL,
    expected_revision BIGINT NOT NULL CHECK (expected_revision >= 0),
    turn_sequence BIGINT CHECK (turn_sequence >= 1),
    status TEXT NOT NULL CHECK (status IN ('queued', 'running', 'succeeded', 'failed', 'unknown')),
    result_refs JSONB NOT NULL DEFAULT '[]'::jsonb CHECK (jsonb_typeof(result_refs) = 'array'),
    usage JSONB NOT NULL DEFAULT '{}'::jsonb CHECK (jsonb_typeof(usage) = 'object'),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (session_id, request_id),
    UNIQUE (session_id, turn_id),
    UNIQUE (session_id, turn_sequence)
);

CREATE TABLE object_versions (
    session_id UUID NOT NULL,
    object_id UUID NOT NULL,
    version INTEGER NOT NULL CHECK (version >= 1),
    source_turn_id UUID NOT NULL,
    kind TEXT NOT NULL,
    conditions JSONB NOT NULL DEFAULT '{}'::jsonb CHECK (jsonb_typeof(conditions) = 'object'),
    source_excerpt TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (session_id, object_id, version),
    FOREIGN KEY (session_id, source_turn_id) REFERENCES turn_requests(session_id, turn_id)
);

CREATE TABLE question_versions (
    session_id UUID NOT NULL,
    question_id UUID NOT NULL,
    question_no INTEGER NOT NULL CHECK (question_no >= 1),
    version INTEGER NOT NULL CHECK (version >= 1),
    source_turn_id UUID NOT NULL,
    question_text TEXT NOT NULL,
    requirements JSONB NOT NULL DEFAULT '[]'::jsonb CHECK (jsonb_typeof(requirements) = 'array'),
    status TEXT NOT NULL CHECK (status IN
        ('ready', 'waiting_input', 'answered', 'partial', 'no_evidence', 'needs_refresh', 'cancelled')),
    relation JSONB,
    object_refs JSONB NOT NULL DEFAULT '[]'::jsonb CHECK (jsonb_typeof(object_refs) = 'array'),
    dependencies JSONB NOT NULL DEFAULT '[]'::jsonb CHECK (jsonb_typeof(dependencies) = 'array'),
    missing_inputs JSONB NOT NULL DEFAULT '[]'::jsonb CHECK (jsonb_typeof(missing_inputs) = 'array'),
    evidence_refs JSONB NOT NULL DEFAULT '[]'::jsonb CHECK (jsonb_typeof(evidence_refs) = 'array'),
    answer_hash CHAR(64),
    source_excerpt TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (session_id, question_id, version),
    FOREIGN KEY (session_id, source_turn_id) REFERENCES turn_requests(session_id, turn_id)
);

CREATE UNIQUE INDEX question_versions_first_number_idx
    ON question_versions (session_id, question_no) WHERE version = 1;
CREATE INDEX question_versions_number_idx
    ON question_versions (session_id, question_no, version DESC);
