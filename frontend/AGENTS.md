<!-- BEGIN:nextjs-agent-rules -->
# This is NOT the Next.js you know

This version has breaking changes — APIs, conventions, and file structure may all differ from your training data. Read the relevant guide in `node_modules/next/dist/docs/` before writing any code. Heed deprecation notices.
<!-- END:nextjs-agent-rules -->

## API Architecture

Use tRPC only for application backend calls. Do not add `api` route folders.
The tRPC HTTP adapter lives at `src/app/trpc/[trpc]/route.ts`, which serves
`/trpc`. Add new backend procedures under `src/server/trpc/routers` and
register them in `src/server/trpc/root.ts`.

## UI Architecture

Use Tailwind CSS v4 utilities for styling throughout the app. Use shadcn/ui as
the default component pattern for reusable UI primitives, keeping theme tokens in
`src/app/globals.css`, component source under `src/components/ui`, and shared UI
helpers under `src/lib`.
