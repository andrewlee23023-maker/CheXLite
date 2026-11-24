// node llm/test_parity.mjs: app/note.js must agree with llm/note_rules.py on every vector.
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const src = readFileSync(join(here, "..", "app", "note.js"), "utf8");
const { verify, templateNote } = await import("data:text/javascript," + encodeURIComponent(src));
const vecs = JSON.parse(readFileSync(join(here, "rule_vectors.json"), "utf8"));

let bad = 0;
for (const v of vecs) {
  const got = verify(v.note, v.facts).ok;
  if (got !== v.expect_ok) {
    bad++;
    if (bad <= 5) console.log(`MISMATCH [${v.kind}] js=${got} py=${v.expect_ok}\n  ${v.note}\n  ${verify(v.note, v.facts).problems}`);
  }
  if (v.template && templateNote(v.facts) !== v.template) {
    bad++;
    if (bad <= 5) console.log(`TEMPLATE MISMATCH\n  js: ${templateNote(v.facts)}\n  py: ${v.template}`);
  }
}
console.log(`${vecs.length} vectors, ${bad} mismatches`);
process.exit(bad ? 1 : 0);
