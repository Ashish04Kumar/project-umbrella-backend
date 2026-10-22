-- Full-day reminders: a reminder for a whole day, with no time of its own ("Monday full day: Riya's exam").
-- due_at holds the morning moment it is said out loud (9:00 local by default).
ALTER TABLE reminders ADD COLUMN IF NOT EXISTS all_day boolean NOT NULL DEFAULT false;
