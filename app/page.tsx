// Skip static prerendering — the page mounts a client component that
// imports @huggingface/transformers (browser-only).
export const dynamic = "force-dynamic";
export { default } from "./VectorSpaceApp";