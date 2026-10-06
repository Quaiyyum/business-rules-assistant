"""Download the public source PDFs into docs/raw/.

Usage:
    python src/fetch_docs.py            # download anything missing
    python src/fetch_docs.py --force    # re-download everything

docs/raw/ is gitignored, so this script is how a fresh clone gets its documents.
Source details (publisher, edition, what each is used for) are in docs/sources.md.
"""

import sys
from pathlib import Path

import requests
from pypdf import PdfReader

RAW_DIR = Path(__file__).resolve().parent.parent / "docs" / "raw"

# (filename, domain, url)
SOURCES = [
    # Light vehicles
    ("nzta_safe_loading_towing_light.pdf", "light",
     "https://www.nzta.govt.nz/assets/resources/glovebox-guide-safe-loading-towing/docs/guide-safe-loading-towing.pdf"),
    ("nzta_virm_nlv_allowable_modifications.pdf", "light",
     "https://vehicleinspection.nzta.govt.nz/downloadPdf/285"),
    ("lvvta_certification_threshold_schedule.pdf", "light",
     "https://www.lvvta.org.nz/documents/suplementary_information/LVVTA_LVV_Cert_Threshold.pdf"),
    # Heavy vehicles
    ("fuso_canter_brake_wheelbase_compliance.pdf", "heavy",
     "https://www.fuso.co.nz/sites/default/files/inline-files/FUSO24_Brake_Data_Sheet_Canter_Combined_1.1_0.pdf"),
    ("fuso_fighter_euro5_brake_wheelbase_compliance.pdf", "heavy",
     "https://www.fuso.co.nz/sites/default/files/inline-files/Fuso%20Fighter%20Euro%205%20Brake_%26%20Wheelbase%20Statement%20of%20Compliance%20Jan%202022_1.pdf"),
    ("fuso_fighter_euro6_wheelbase_range.pdf", "heavy",
     "https://www.fuso.co.nz/sites/default/files/2026-05/Fighter%20Euro%206%20Wheelbase%20Range%20Chart%20%20Updated%20June%202026.pdf"),
    ("nzta_truck_loading_code_2026.pdf", "heavy",
     "https://www.nzta.govt.nz/assets/consultation/updates-to-the-official-nz-truck-loading-code/truck-loading-code-revised-edition.pdf"),
    ("nzta_virm_hvsc.pdf", "heavy",
     "https://vehicleinspection.nzta.govt.nz/downloadPdf/50"),
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    ),
    "Accept": "application/pdf,*/*",
}


def fetch(filename: str, url: str, force: bool) -> str:
    target = RAW_DIR / filename
    if target.exists() and not force:
        return f"skip   {filename} (already downloaded)"
    try:
        response = requests.get(url, headers=HEADERS, timeout=120)
        response.raise_for_status()
    except requests.RequestException as error:
        return f"FAIL   {filename}: {error}"
    # The PDF spec allows the %PDF marker anywhere in the first 1024 bytes.
    if b"%PDF" not in response.content[:1024]:
        kind = response.headers.get("Content-Type", "unknown type")
        return (
            f"FAIL   {filename}: the URL did not return a PDF "
            f"(got {kind}, {len(response.content)} bytes, final URL {response.url}, "
            f"starts {response.content[:80]!r})"
        )
    target.write_bytes(response.content)
    try:
        pages = len(PdfReader(target).pages)
    except Exception as error:  # a download that will not open is a failure
        target.unlink()
        return f"FAIL   {filename}: PDF will not open ({error})"
    return f"ok     {filename} ({pages} pages, {len(response.content) // 1024} KB)"


def main() -> int:
    force = "--force" in sys.argv
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    results = [fetch(name, url, force) for name, _domain, url in SOURCES]
    print("\n".join(results))
    failed = sum(line.startswith("FAIL") for line in results)
    print(f"\n{len(results) - failed} of {len(results)} sources available in {RAW_DIR}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
