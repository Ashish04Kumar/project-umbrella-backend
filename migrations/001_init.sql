-- Companion umbrella, first tables. Runs inside its own schema (search_path is set by the app).
-- user_id is a plain string chosen by the body (the Synq backend passes its own user id).

CREATE TABLE companion_users (
    id text PRIMARY KEY,
    display_name text NOT NULL DEFAULT '',
    timezone text NOT NULL DEFAULT 'Asia/Kolkata',
    settings jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE conversations (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id text NOT NULL,
    started_at timestamptz NOT NULL DEFAULT now(),
    last_message_at timestamptz NOT NULL DEFAULT now(),
    summary text NOT NULL DEFAULT '',
    summarized_count integer NOT NULL DEFAULT 0   -- how many of the oldest messages the summary covers
);
CREATE INDEX conversations_user ON conversations (user_id, last_message_at DESC);

CREATE TABLE messages (
    id bigserial PRIMARY KEY,
    conversation_id uuid NOT NULL REFERENCES conversations (id) ON DELETE CASCADE,
    role text NOT NULL CHECK (role IN ('user', 'assistant', 'tool')),
    content text NOT NULL DEFAULT '',
    expression text,
    tool_calls jsonb,
    tool_call_id text,
    name text,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX messages_conversation ON messages (conversation_id, id);

-- Long-term facts and working memory. kind = fact | preference | person | routine | event.
-- key is a short slug ("food:paneer", "person:riya") so the same thing is updated, not duplicated.
CREATE TABLE memories (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id text NOT NULL,
    kind text NOT NULL,
    key text NOT NULL,
    text text NOT NULL,
    data jsonb NOT NULL DEFAULT '{}'::jsonb,
    confidence real NOT NULL DEFAULT 1,
    status text NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'pending')),
    sensitive boolean NOT NULL DEFAULT false,
    expires_at timestamptz,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    search tsvector GENERATED ALWAYS AS (to_tsvector('simple', replace(key, ':', ' ') || ' ' || text)) STORED
);
CREATE UNIQUE INDEX memories_user_key ON memories (user_id, key, status);
CREATE INDEX memories_search ON memories USING gin (search);

CREATE TABLE reminders (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id text NOT NULL,
    text text NOT NULL,
    due_at timestamptz NOT NULL,
    repeat text NOT NULL DEFAULT 'none' CHECK (repeat IN ('none', 'daily', 'weekly')),
    status text NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'done', 'cancelled')),
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX reminders_due ON reminders (status, due_at);

CREATE TABLE todos (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id text NOT NULL,
    text text NOT NULL,
    due_date date,
    priority text NOT NULL DEFAULT 'normal' CHECK (priority IN ('low', 'normal', 'high')),
    done boolean NOT NULL DEFAULT false,
    created_at timestamptz NOT NULL DEFAULT now(),
    done_at timestamptz
);
CREATE INDEX todos_user ON todos (user_id, done);

CREATE TABLE notes (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id text NOT NULL,
    text text NOT NULL,
    tags text[] NOT NULL DEFAULT '{}',
    created_at timestamptz NOT NULL DEFAULT now(),
    search tsvector GENERATED ALWAYS AS (to_tsvector('simple', text)) STORED
);
CREATE INDEX notes_search ON notes USING gin (search);

CREATE TABLE lists (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id text NOT NULL,
    name text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX lists_user_name ON lists (user_id, lower(name));

CREATE TABLE list_items (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    list_id uuid NOT NULL REFERENCES lists (id) ON DELETE CASCADE,
    text text NOT NULL,
    done boolean NOT NULL DEFAULT false,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE tool_log (
    id bigserial PRIMARY KEY,
    user_id text NOT NULL,
    tool text NOT NULL,
    args jsonb,
    result jsonb,
    ok boolean NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
