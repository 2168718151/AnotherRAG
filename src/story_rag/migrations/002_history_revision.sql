ALTER TABLE object_versions
    ADD COLUMN changes JSONB NOT NULL DEFAULT '[]'::jsonb
        CHECK (jsonb_typeof(changes) = 'array');
ALTER TABLE object_versions
    ADD COLUMN derived_from JSONB;

ALTER TABLE question_versions
    ADD COLUMN revision_kind TEXT
        CHECK (revision_kind IN
            ('supplement', 'replace', 'extend', 'compare', 'format', 'cancel', 'new'));

CREATE INDEX turn_requests_completed_scope_idx
    ON turn_requests (session_id, turn_sequence DESC)
    WHERE status = 'succeeded';
