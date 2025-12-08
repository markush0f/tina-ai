-- ============================================
-- EXTENSIONS
-- ============================================
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ============================================
-- USERS
-- ============================================
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- ============================================
-- CONTENT ITEMS (BASE TABLE)
-- ============================================
CREATE TABLE content_items (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,

    type TEXT NOT NULL,
    source TEXT,

    event_timestamp TIMESTAMP NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_content_items_user ON content_items(user_id);
CREATE INDEX idx_content_items_event_ts ON content_items(event_timestamp);

-- ============================================
-- TEXT CONTENT
-- ============================================
CREATE TABLE content_text (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content_item_id UUID NOT NULL UNIQUE REFERENCES content_items(id) ON DELETE CASCADE,
    text_content TEXT NOT NULL
);

-- ============================================
-- MEDIA CONTENT
-- ============================================
CREATE TABLE content_media (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content_item_id UUID NOT NULL UNIQUE REFERENCES content_items(id) ON DELETE CASCADE,
    media_type TEXT NOT NULL,
    storage_path TEXT NOT NULL,
    extracted_text TEXT,
    duration_seconds INT,
    metadata JSONB
);

CREATE INDEX idx_media_type ON content_media(media_type);

-- ============================================
-- LINKS CONTENT
-- ============================================
CREATE TABLE content_links (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content_item_id UUID NOT NULL UNIQUE REFERENCES content_items(id) ON DELETE CASCADE,
    url TEXT NOT NULL,
    metadata JSONB,
    fetched_text TEXT
);

-- ============================================
-- CONTENT ANALYSIS (AI OUTPUT PER ITEM)
-- ============================================
CREATE TABLE content_analysis (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content_item_id UUID NOT NULL UNIQUE REFERENCES content_items(id) ON DELETE CASCADE,

    sentiment TEXT,
    emotion TEXT,
    topics JSONB,
    entities JSONB,
    importance_score NUMERIC,

    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_analysis_topics ON content_analysis USING GIN(topics);
CREATE INDEX idx_analysis_entities ON content_analysis USING GIN(entities);

-- ============================================
-- DAILY SUMMARIES
-- ============================================
CREATE TABLE daily_summaries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,

    year INT NOT NULL,
    month INT NOT NULL,
    day INT NOT NULL,

    summary_text TEXT NOT NULL,
    highlights JSONB,
    stats JSONB,

    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),

    UNIQUE(user_id, year, month, day)
);

CREATE INDEX idx_daily_user ON daily_summaries(user_id);
CREATE INDEX idx_daily_date ON daily_summaries(year, month, day);

-- ============================================
-- WEEKLY SUMMARIES (derived from daily_summaries)
-- ============================================
CREATE TABLE weekly_summaries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,

    year INT NOT NULL,
    week_number INT NOT NULL,          -- ISO week number (1–53)

    summary_text TEXT NOT NULL,
    highlights JSONB,
    stats JSONB,

    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),

    UNIQUE(user_id, year, week_number)
);

CREATE INDEX idx_weekly_user ON weekly_summaries(user_id);
CREATE INDEX idx_weekly_year ON weekly_summaries(year, week_number);

-- ============================================
-- MONTHLY SUMMARIES
-- ============================================
CREATE TABLE monthly_summaries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,

    year INT NOT NULL,
    month INT NOT NULL,

    summary_text TEXT NOT NULL,
    highlights JSONB,
    stats JSONB,

    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),

    UNIQUE(user_id, year, month)
);

CREATE INDEX idx_monthly_user ON monthly_summaries(user_id);

-- ============================================
-- YEARLY SUMMARIES
-- ============================================
CREATE TABLE yearly_summaries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,

    year INT NOT NULL,

    summary_text TEXT NOT NULL,
    highlights JSONB,
    stats JSONB,

    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),

    UNIQUE(user_id, year)
);

CREATE INDEX idx_yearly_user ON yearly_summaries(user_id);

-- ============================================
-- EVENT COUNTERS (habits / repeated behaviors)
-- ============================================
CREATE TABLE event_counters (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,

    event_name TEXT NOT NULL,
    year INT NOT NULL,
    month INT,

    count INT NOT NULL DEFAULT 0,

    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),

    UNIQUE (user_id, event_name, year, month)
);

CREATE INDEX idx_event_counters_user ON event_counters(user_id);
CREATE INDEX idx_event_counters_event ON event_counters(event_name);
CREATE INDEX idx_event_counters_year ON event_counters(year);
