-- Threads: each conversation belongs to a thread. "main" is the user's own ongoing chat with the
-- companion; it is never reset or replaced. "New conversation" opens a separate thread.
ALTER TABLE conversations ADD COLUMN thread text NOT NULL DEFAULT 'main';
CREATE INDEX conversations_thread ON conversations (user_id, thread, last_message_at DESC);
