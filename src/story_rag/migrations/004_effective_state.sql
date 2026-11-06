ALTER TABLE question_versions
    ADD COLUMN effective_state JSONB NOT NULL DEFAULT '{}'::jsonb
        CHECK (jsonb_typeof(effective_state) = 'object');

ALTER TABLE turn_requests
    ADD COLUMN execution_owner JSONB;
