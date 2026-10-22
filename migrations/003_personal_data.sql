-- Personal data v2 (docs/personal-data-spec.md, docs/search-and-retrieval-spec.md):
-- short/long-term memory with structured fields, titled notes, day-wise to-dos, richer reminders,
-- weighted full-text search kept by triggers, and fuzzy (trigram) matching.

CREATE EXTENSION IF NOT EXISTS pg_trgm SCHEMA public;

-- Memory --------------------------------------------------------------------------------------------
ALTER TABLE memories DROP COLUMN search;
ALTER TABLE memories
    ADD COLUMN scope text NOT NULL DEFAULT 'long' CHECK (scope IN ('short', 'long')),
    ADD COLUMN category text NOT NULL DEFAULT 'other',
    ADD COLUMN subject text NOT NULL DEFAULT '',
    ADD COLUMN topic text NOT NULL DEFAULT '',
    ADD COLUMN event_date date,                        -- the day it is about (short) or a date fact (long)
    ADD COLUMN yearly boolean NOT NULL DEFAULT false,  -- birthdays, anniversaries: the year does not matter
    ADD COLUMN keywords text[] NOT NULL DEFAULT '{}',
    ADD COLUMN entities text[] NOT NULL DEFAULT '{}',
    ADD COLUMN pinned boolean NOT NULL DEFAULT false,  -- profile facts that go with every message
    ADD COLUMN search tsvector;
UPDATE memories SET
    scope = CASE WHEN kind = 'event' THEN 'short' ELSE 'long' END,
    category = CASE kind WHEN 'fact' THEN 'profile' WHEN 'preference' THEN 'preference'
                         WHEN 'person' THEN 'person' WHEN 'event' THEN 'event' ELSE 'other' END;

-- Weighted search text: A = subject/topic/title, B = keywords and entities, C = the text itself.
CREATE FUNCTION memories_search_update() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    NEW.search :=
        setweight(to_tsvector('simple', coalesce(NEW.subject, '') || ' ' || coalesce(NEW.topic, '') || ' ' || replace(coalesce(NEW.key, ''), ':', ' ')), 'A') ||
        setweight(to_tsvector('simple', array_to_string(NEW.keywords || NEW.entities, ' ')), 'B') ||
        setweight(to_tsvector('simple', coalesce(NEW.text, '')), 'C');
    RETURN NEW;
END $$;
CREATE TRIGGER memories_search BEFORE INSERT OR UPDATE ON memories FOR EACH ROW EXECUTE FUNCTION memories_search_update();
UPDATE memories SET text = text;
CREATE INDEX memories_search ON memories USING gin (search);
CREATE INDEX memories_text_trgm ON memories USING gin (text public.gin_trgm_ops);
CREATE INDEX memories_subject_trgm ON memories USING gin (subject public.gin_trgm_ops);
CREATE INDEX memories_entities ON memories USING gin (entities);
CREATE INDEX memories_scope ON memories (user_id, scope, expires_at);

-- Notes ---------------------------------------------------------------------------------------------
ALTER TABLE notes DROP COLUMN search;
ALTER TABLE notes
    ADD COLUMN title text NOT NULL DEFAULT '',
    ADD COLUMN keywords text[] NOT NULL DEFAULT '{}',
    ADD COLUMN entities text[] NOT NULL DEFAULT '{}',
    ADD COLUMN search tsvector;
CREATE FUNCTION notes_search_update() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    NEW.search :=
        setweight(to_tsvector('simple', coalesce(NEW.title, '')), 'A') ||
        setweight(to_tsvector('simple', array_to_string(NEW.keywords || NEW.entities || NEW.tags, ' ')), 'B') ||
        setweight(to_tsvector('simple', coalesce(NEW.text, '')), 'C');
    RETURN NEW;
END $$;
CREATE TRIGGER notes_search BEFORE INSERT OR UPDATE ON notes FOR EACH ROW EXECUTE FUNCTION notes_search_update();
UPDATE notes SET text = text;
CREATE INDEX notes_search ON notes USING gin (search);
CREATE INDEX notes_text_trgm ON notes USING gin (text public.gin_trgm_ops);
CREATE INDEX notes_title_trgm ON notes USING gin (title public.gin_trgm_ops);
CREATE INDEX notes_entities ON notes USING gin (entities);

-- To-dos: every to-do belongs to a day ---------------------------------------------------------------
ALTER TABLE todos ADD COLUMN day date;
UPDATE todos SET day = coalesce(due_date, created_at::date);
ALTER TABLE todos ALTER COLUMN day SET NOT NULL, ALTER COLUMN day SET DEFAULT current_date;
ALTER TABLE todos DROP COLUMN due_date;
ALTER TABLE todos ADD COLUMN entities text[] NOT NULL DEFAULT '{}';
CREATE INDEX todos_day ON todos (user_id, day, done);
CREATE INDEX todos_text_trgm ON todos USING gin (text public.gin_trgm_ops);

-- Reminders: monthly repeat, fired/snoozed, spoken once more if not answered -------------------------
ALTER TABLE reminders DROP CONSTRAINT reminders_repeat_check;
ALTER TABLE reminders ADD CONSTRAINT reminders_repeat_check CHECK (repeat IN ('none', 'daily', 'weekly', 'monthly'));
ALTER TABLE reminders DROP CONSTRAINT reminders_status_check;
ALTER TABLE reminders ADD CONSTRAINT reminders_status_check CHECK (status IN ('pending', 'fired', 'snoozed', 'done', 'cancelled'));
ALTER TABLE reminders
    ADD COLUMN fired_at timestamptz,
    ADD COLUMN reminded_count integer NOT NULL DEFAULT 0,
    ADD COLUMN entities text[] NOT NULL DEFAULT '{}';
CREATE INDEX reminders_user_due ON reminders (user_id, status, due_at);
CREATE INDEX reminders_text_trgm ON reminders USING gin (text public.gin_trgm_ops);

-- Lists: fuzzy list names ("shoping") and item text ----------------------------------------------------
CREATE INDEX lists_name_trgm ON lists USING gin (name public.gin_trgm_ops);
CREATE INDEX list_items_text_trgm ON list_items USING gin (text public.gin_trgm_ops);
