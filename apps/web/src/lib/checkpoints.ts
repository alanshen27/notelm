import type { Checkpoint } from "./api";

const TIMESTAMP_PT = /^\d{8}-\d{6}\.pt$/i;

/** Checkpoint filenames from 2026-08-29 on. Product names (Prelude, …) stay on the site. */
export const CHECKPOINT_CODENAMES = [
  "invention",
  "etude",
  "prelude",
  "chaconne",
  "canon",
  "sinfonia",
] as const;

/** Hidden on the public picker. /dev/ still lists them by code name. */
export const EXPERIMENTAL_CODENAMES = ["invention", "etude"] as const;

const CODENAME_SET = new Set<string>(CHECKPOINT_CODENAMES);
const EXPERIMENTAL_SET = new Set<string>(EXPERIMENTAL_CODENAMES);

function stemOf(name: string) {
  return String(name || "").replace(/\.pt$/i, "").toLowerCase();
}

export function isNamedCheckpoint(name: string) {
  const n = String(name || "").toLowerCase();
  if (!n.endsWith(".pt") || TIMESTAMP_PT.test(n)) return false;
  return true;
}

export function isCodeNameCheckpoint(name: string) {
  return CODENAME_SET.has(stemOf(name));
}

export function isExperimentalCheckpoint(name: string) {
  return EXPERIMENTAL_SET.has(stemOf(name));
}

export function checkpointLabel(c: Checkpoint, opts?: { verbose?: boolean }) {
  const stem = String(c.name || "").replace(/\.pt$/i, "");
  let label = stem;
  if (stem === "weights") label = "weights (default)";
  else if (!isNamedCheckpoint(c.name)) {
    const tag = [c.model, c.tokenizer].filter(Boolean).join("/");
    const tail = [c.parent, c.name].filter(Boolean).join("/");
    label = tag ? `${tag} · ${tail}` : tail || stem;
  }
  if (opts?.verbose && c.tokenizer && !label.includes(String(c.tokenizer))) {
    label = `${label} · ${c.tokenizer}`;
  }
  return label;
}

export function matchCheckpoint(list: Checkpoint[], query: string | null | undefined) {
  const q = String(query || "").trim().toLowerCase();
  if (!q || !list?.length) return "";
  if (q === "prelude") return preferCheckpoint(list);
  const exact = list.find((c) => {
    const name = String(c.name || "").toLowerCase();
    const stem = name.replace(/\.pt$/i, "");
    return stem === q || name === q || name === `${q}.pt`;
  });
  if (exact) return exact.path;
  const hit = list.find((c) => {
    const hay = [c.name, c.path, c.tokenizer, c.parent, c.model]
      .filter(Boolean)
      .join(" ")
      .toLowerCase();
    return hay.includes(q);
  });
  return hit?.path ?? "";
}

export function cowriterCheckpoints(list: Checkpoint[]) {
  const coded = (list || []).filter(
    (c) => isCodeNameCheckpoint(c.name) && !isExperimentalCheckpoint(c.name)
  );
  return [...coded].sort((a, b) =>
    checkpointLabel(a).localeCompare(checkpointLabel(b), undefined, { numeric: true })
  );
}

export function preferCheckpoint(list: Checkpoint[]) {
  const publicList = cowriterCheckpoints(list);
  const hit =
    publicList.find((c) => c.name === "prelude.pt") || publicList[0];
  return hit?.path ?? "";
}
