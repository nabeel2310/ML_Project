-- Initial schema for crop disease prototype.
-- RLS is enabled and no broad public policies are created here.
-- Review/adjust policies after the app's authentication model is decided.

create extension if not exists pgcrypto;

create table if not exists public.crop_diseases (
    id uuid primary key default gen_random_uuid(),
    crop text not null check (crop in ('Coconut', 'Arecanut', 'Banana', 'Paddy', 'Pepper', 'Cardamom')),
    label text not null,
    description text,
    symptoms text[] not null default '{}',
    recommended_actions text[] not null default '{}',
    is_healthy boolean not null default false,
    created_at timestamptz not null default now(),
    unique (crop, label)
);

create table if not exists public.training_samples (
    id uuid primary key default gen_random_uuid(),
    image_path text not null,
    crop text not null check (crop in ('Coconut', 'Arecanut', 'Banana', 'Paddy', 'Pepper', 'Cardamom')),
    disease_label text not null,
    symptom_text text,
    source_dataset text,
    group_id text,
    reviewed boolean not null default false,
    created_at timestamptz not null default now()
);

create table if not exists public.prediction_logs (
    id uuid primary key default gen_random_uuid(),
    crop text,
    predicted_label text,
    confidence double precision check (confidence is null or (confidence >= 0 and confidence <= 1)),
    model_version text,
    symptom_text text,
    created_at timestamptz not null default now()
);

alter table public.crop_diseases enable row level security;
alter table public.training_samples enable row level security;
alter table public.prediction_logs enable row level security;

-- Intentionally no anon/authenticated policies yet:
-- this prevents public reads/writes until access requirements are explicitly defined.
-- Use server-side service-role access only in trusted backend code; never expose the key.
