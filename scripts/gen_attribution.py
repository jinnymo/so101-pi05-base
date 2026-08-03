# SPDX-License-Identifier: Apache-2.0
"""Generate ATTRIBUTION.md for the released SO-101 / SO-100 dataset.

Inputs:
  repack_plan.json   include/exclude dataset keys
  license_join.json  per-dataset license resolved from the crawl catalog
  27_manifest.json   per-dataset episode drops applied during the merge
  dataset tree       episode parquet files, used to count episodes

Episode count per source = parquet files on disk - episodes dropped by the merge.

Usage:
    python gen_attribution.py --src /path/to/base_train
"""

import argparse
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PLAN = f"{HERE}/repack_plan.json"
LICENSE_JOIN = f"{HERE}/license_join.json"
DEFAULT_MANIFEST = os.path.normpath(f"{HERE}/../pipeline/artifacts/27_manifest.json")
DEFAULT_OUT = os.path.normpath(f"{HERE}/../dataset/ATTRIBUTION.md")

AUTHOR = "Dongyoon Kim"

NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
                7: "seven", 8: "eight", 9: "nine", 10: "ten"}

MIT_NOTICE = [
    'Permission is hereby granted, free of charge, to any person obtaining a copy of '
    'this software and associated documentation files (the "Software"), to deal in the '
    "Software without restriction, including without limitation the rights to use, copy, "
    "modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, "
    "and to permit persons to whom the Software is furnished to do so, subject to the "
    "following conditions:",
    "The above copyright notice and this permission notice shall be included in all "
    "copies or substantial portions of the Software.",
    'THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, '
    "INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A "
    "PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT "
    "HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION "
    "OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OR IN CONNECTION WITH THE SOFTWARE OR "
    "THE USE OR OTHER DEALINGS IN THE SOFTWARE.",
]


def spell(n):
    return NUMBER_WORDS.get(n, str(n))


def repo_id(key):
    """external/tier1/owner__name -> owner/name. self/name -> None."""
    head, _, last = key.rpartition("/")
    if key.startswith("self/"):
        return None
    return last.replace("__", "/", 1)


def episode_count(src, key, drop_eps):
    files = glob.glob(os.path.join(src, key, "data", "chunk-*", "episode_*.parquet"))
    return max(0, len(files) - len(drop_eps))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", required=True,
                    help="directory holding the per-source datasets "
                         "(external/tier*/<owner>__<name> and self/<name>)")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST,
                    help="27_manifest.json (default: ../pipeline/artifacts/27_manifest.json)")
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help="output file (default: ../dataset/ATTRIBUTION.md)")
    args = ap.parse_args()

    plan = json.load(open(PLAN))
    lic = {r["key"]: r for r in json.load(open(LICENSE_JOIN))["keep"]}
    manifest = json.load(open(args.manifest))["datasets"]

    external, own = [], []
    for key in plan["include"]:
        drop_eps = manifest.get(key, {}).get("drop_eps") or []
        row = {
            "key": key,
            "repo": repo_id(key),
            "license": lic[key]["lic"],
            "episodes": episode_count(args.src, key, drop_eps),
        }
        (own if row["repo"] is None else external).append(row)

    external.sort(key=lambda r: r["repo"].lower())
    own.sort(key=lambda r: r["key"])

    non_apache = [r for r in external if r["license"] != "apache-2.0"]
    if len(non_apache) != 1 or non_apache[0]["license"] != "mit":
        raise SystemExit("the Licenses section is written for exactly one MIT source; "
                         "the license mix changed and the prose needs rewriting by hand")
    mit_row = non_apache[0]

    n_apache = sum(1 for r in external if r["license"] == "apache-2.0")
    ep_apache = sum(r["episodes"] for r in external if r["license"] == "apache-2.0")
    ep_mit = mit_row["episodes"]
    ep_own = sum(r["episodes"] for r in own)
    total_ds = len(external) + len(own)
    total_ep = ep_apache + ep_mit + ep_own

    lines = []
    w = lines.append
    w("# Attribution")
    w("")
    w(
        "This dataset redistributes work created by other people. This file is the "
        "attribution notice required by Section 4 of the Apache License 2.0, and it "
        "records the provenance of every episode in the release."
    )
    w("")
    w(f"- {total_ds} source datasets, {total_ep:,} episodes")
    w(f"- apache-2.0, from the Hugging Face Hub: {n_apache} datasets, {ep_apache:,} episodes")
    w(f"- mit, from the Hugging Face Hub: 1 dataset, {ep_mit:,} episodes")
    w(f"- apache-2.0, recorded by {AUTHOR}: {len(own)} datasets, {ep_own:,} episodes")
    w("")
    w(
        "The episode count is the number of episodes taken from that source after "
        "deduplication and per-episode drops. It can be lower than the episode count "
        "of the upstream repository. Licenses were read from the upstream repository "
        "metadata at collection time; upstream repositories can change, and the "
        "upstream entry is authoritative."
    )
    w("")
    w("## Licenses")
    w("")
    w(
        "A copy of the Apache License 2.0 is distributed with this dataset as "
        "`LICENSE`, as Section 4(a) of that license requires."
    )
    w("")
    w(
        f"The {spell(len(own))} sources recorded by {AUTHOR} are released under the "
        'Apache License 2.0. "Recorded by the author" is a provenance category, not a '
        "license."
    )
    w("")
    w(
        f"One source, [{mit_row['repo']}]"
        f"(https://huggingface.co/datasets/{mit_row['repo']}) "
        f"({mit_row['episodes']} episodes), is MIT rather than apache-2.0. MIT permits "
        "use, copying, modification and redistribution provided its copyright notice "
        "and permission notice are included in all copies. The upstream repository "
        "declares `license: mit` in its dataset card metadata and ships no `LICENSE` "
        "file and no copyright line, so there is no upstream notice text to reproduce "
        "verbatim; this entry and the table row below are the attribution. The MIT "
        "permission notice, in its standard form, reads:"
    )
    w("")
    for i, para in enumerate(MIT_NOTICE):
        if i:
            w(">")
        w(f"> {para}")
    w("")
    w("## Modifications")
    w("")
    w(
        "Section 4(b) of the Apache License 2.0 requires stating that files were "
        "changed. Every source dataset was modified as follows before inclusion:"
    )
    w("")
    w("- datasets published in LeRobot v3.0 format were converted to v2.1")
    w(
        "- camera streams were remapped onto three fixed slots (`base_0_rgb`, "
        "`left_wrist_0_rgb`, `right_wrist_0_rgb`); a slot left empty by the keyword "
        "pass was filled from whatever cameras remained, slots still empty after that "
        "were filled with black placeholder video, and a per-slot mask column was added"
    )
    w("- depth, infrared and surplus camera streams were dropped")
    w(
        "- episode, frame and task indices were renumbered into a single global "
        "index space, and the per-source metadata files were regenerated"
    )
    w(
        "- a small number of task strings were rewritten into plain English "
        "instructions (integer placeholders, dict reprs, snake_case, one "
        "non-English source)"
    )
    w("- individual episodes were dropped where they were empty, truncated or corrupt")
    w("")
    w("No action, state or image content was otherwise altered.")
    w("")
    w("## External sources")
    w("")
    w("| Source | License | Episodes |")
    w("|---|---|---|")
    for r in external:
        w(
            f"| [{r['repo']}](https://huggingface.co/datasets/{r['repo']}) "
            f"| {r['license']} | {r['episodes']} |"
        )
    w("")
    w("## Sources recorded by the author")
    w("")
    w(
        f"Recorded by {AUTHOR} on SO-101 hardware and released here under the "
        "Apache License 2.0."
    )
    w("")
    w("| Source | License | Episodes |")
    w("|---|---|---|")
    for r in own:
        name = r["key"].split("/", 1)[1]
        w(f"| {name} | apache-2.0 | {r['episodes']} |")
    w("")
    w("## Not redistributed")
    w("")
    w(
        f"{len(plan['exclude'])} further datasets ({plan['exclude_ep']:,} episodes) "
        "were used to train the model that accompanies this release but are not "
        "redistributed here, because their upstream repositories declare no license."
    )
    w("")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"{args.out}: {total_ds} sources ({len(external)} external + {len(own)} own), "
          f"{total_ep:,} episodes")


if __name__ == "__main__":
    main()
