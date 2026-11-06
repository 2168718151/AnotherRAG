ALTER TABLE question_versions DROP CONSTRAINT question_versions_status_check;
ALTER TABLE question_versions ADD CONSTRAINT question_versions_status_check
    CHECK (status IN ('ready', 'waiting_input', 'answered', 'partial',
        'no_evidence', 'needs_refresh', 'cancelled', 'technical_failure'));

CREATE TABLE history_index_outbox (
    session_id UUID NOT NULL REFERENCES conversation_sessions(session_id) ON DELETE CASCADE,
    turn_id UUID NOT NULL,
    turn_sequence BIGINT NOT NULL CHECK (turn_sequence >= 1),
    processed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (session_id, turn_id)
);
CREATE INDEX history_index_outbox_pending_idx
    ON history_index_outbox (session_id, turn_sequence) WHERE processed_at IS NULL;

CREATE TABLE history_question_index (
    session_id UUID NOT NULL REFERENCES conversation_sessions(session_id) ON DELETE CASCADE,
    question_id UUID NOT NULL,
    question_no INTEGER NOT NULL,
    version INTEGER NOT NULL,
    question_text TEXT NOT NULL,
    status TEXT NOT NULL,
    source_turn_id UUID NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (session_id, question_id)
);
