CREATE TABLE conversation_display_turns (
    session_id UUID NOT NULL,
    turn_id UUID NOT NULL,
    user_message TEXT NOT NULL,
    assistant_answer TEXT,
    created_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (session_id, turn_id),
    FOREIGN KEY (session_id, turn_id)
        REFERENCES turn_requests (session_id, turn_id) ON DELETE CASCADE
);

CREATE INDEX conversation_display_turns_page_idx
    ON conversation_display_turns (session_id, created_at, turn_id);

CREATE INDEX turn_requests_display_page_idx
    ON turn_requests (session_id, created_at, turn_id);
