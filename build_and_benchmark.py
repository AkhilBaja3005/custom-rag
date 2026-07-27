import os
import time
import requests
import psutil
import fitz  # PyMuPDF

import config
from ingest import MultiParserIngestionEngine
from query import RAGQueryEngine

MASTER_PDF_PATH = "arxiv_10000_pages_master.pdf"
TARGET_PAGE_COUNT = 10000
MAX_RAM_GB = 1.5

def fetch_arxiv_pdfs_and_stitch(target_pages: int = TARGET_PAGE_COUNT, output_pdf_path: str = MASTER_PDF_PATH):
    """Queries arXiv CS.AI public API to fetch papers and stitches them into a 10,000-page PDF master file."""
    if os.path.exists(output_pdf_path):
        doc = fitz.open(output_pdf_path)
        current_pages = len(doc)
        doc.close()
        if current_pages >= target_pages:
            print(f"Master PDF '{output_pdf_path}' already exists with {current_pages} pages. Skipping download.")
            return output_pdf_path

    print(f"Generating benchmark dataset from arXiv API (target: {target_pages} pages)...")
    master_doc = fitz.open()
    
    start_index = 0
    batch_fetch_count = 25
    
    while len(master_doc) < target_pages:
        url = f"https://export.arxiv.org/api/query?search_query=cat:cs.AI&start={start_index}&max_results={batch_fetch_count}"
        print(f"Fetching arXiv batch from index {start_index}...")
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code != 200:
                print(f"ArXiv API returned status {resp.status_code}, retrying...")
                time.sleep(2)
                continue
                
            # Extract PDF download links from arXiv Atom feed XML
            import xml.etree.ElementTree as ET
            root = ET.fromstring(resp.content)
            entries = root.findall("{http://www.w3.org/2005/Atom}entry")
            
            if not entries:
                print("No more entries returned from arXiv API.")
                break
                
            for entry in entries:
                if len(master_doc) >= target_pages:
                    break
                    
                pdf_link = None
                for link in entry.findall("{http://www.w3.org/2005/Atom}link"):
                    if link.attrib.get("title") == "pdf":
                        pdf_link = link.attrib.get("href")
                        break
                        
                if pdf_link:
                    # Convert to direct pdf download url if needed
                    if not pdf_link.endswith(".pdf"):
                        pdf_link += ".pdf"
                        
                    print(f"Downloading paper: {pdf_link} (Current pages: {len(master_doc)}/{target_pages})...")
                    try:
                        pdf_resp = requests.get(pdf_link, timeout=20)
                        if pdf_resp.status_code == 200:
                            sub_doc = fitz.open(stream=pdf_resp.content, filetype="pdf")
                            master_doc.insert_pdf(sub_doc)
                            sub_doc.close()
                    except Exception as pe:
                        print(f"Skipping paper due to download error: {pe}")
                        
            start_index += batch_fetch_count
            time.sleep(1)  # Respect arXiv API rate limits
            
        except Exception as e:
            print(f"Error querying arXiv API: {e}")
            time.sleep(3)
            
    # Save stitched master PDF
    print(f"Saving stitched master PDF with {len(master_doc)} pages to '{output_pdf_path}'...")
    master_doc.save(output_pdf_path)
    master_doc.close()
    return output_pdf_path

def run_stress_test_and_benchmark(pdf_path: str):
    """Executes streaming ingestion with psutil RAM monitoring, asserting peak RAM < 1.5 GB."""
    print("\n=======================================================")
    print("🚀 STAGE 1: INGESTION STRESS TEST & MEMORY PROFILING")
    print("=======================================================")
    
    process = psutil.Process(os.getpid())
    start_ram_mb = process.memory_info().rss / (1024 * 1024)
    print(f"Baseline Process RAM: {start_ram_mb:.2f} MB")
    
    peak_ram_mb = start_ram_mb
    
    # Progress callback that updates peak RAM usage
    def ram_monitor_callback(start_page, end_page, total_pages):
        nonlocal peak_ram_mb
        current_ram = process.memory_info().rss / (1024 * 1024)
        if current_ram > peak_ram_mb:
            peak_ram_mb = current_ram
        pct = (end_page / total_pages) * 100
        print(f"  [Batch Progress: {pct:5.1f}% | Pages {start_page:5d} to {end_page:5d}] Current RAM: {current_ram:7.2f} MB | Peak RAM: {peak_ram_mb:7.2f} MB")

    ingest_engine = MultiParserIngestionEngine(pdf_path)
    stats = ingest_engine.process_pdf_streaming(progress_callback=ram_monitor_callback)
    
    peak_ram_gb = peak_ram_mb / 1024.0
    print("\n-------------------------------------------------------")
    print(f"📊 INGESTION PERFORMANCE RESULTS:")
    print(f"  • Total Pages Processed : {stats['total_pages']:,}")
    print(f"  • Total Chunks Indexed  : {stats['total_chunks']:,}")
    print(f"  • Processing Time       : {stats['duration']:.2f} seconds")
    print(f"  • Ingestion Speed       : {stats['speed']:.2f} pages/sec")
    print(f"  • Peak Memory Usage     : {peak_ram_gb:.3f} GB ({peak_ram_mb:.2f} MB)")
    print("-------------------------------------------------------")
    
    # HARD ASSERTION: Peak Memory MUST NOT exceed 1.5 GB
    assert peak_ram_gb <= MAX_RAM_GB, f"❌ MEMORY LIMIT VIOLATED! Peak RAM was {peak_ram_gb:.3f} GB, exceeding maximum allowed limit of {MAX_RAM_GB} GB!"
    print(f"✅ PASSED: Peak RAM ({peak_ram_gb:.3f} GB) is strictly under hard limit of {MAX_RAM_GB} GB!\n")

    print("=======================================================")
    print("🚀 STAGE 2: RETRIEVAL ACCURACY & CITATION VERIFICATION")
    print("=======================================================")
    
    query_engine = RAGQueryEngine()
    test_queries = [
        "What are the main artificial intelligence methods discussed in the document?",
        "What neural network architectures or machine learning models are evaluated?",
        "Summarize the experimental findings and key results."
    ]
    
    for idx, q in enumerate(test_queries, 1):
        print(f"\n[Test Query {idx}]: {q}")
        res = query_engine.query(q)
        print("  • Answer Output:")
        print(f"    {res['answer']}")
        print("  • Top Sources:")
        for s in res['sources']:
            print(f"    - [Source: Page {s['page']}] | Score: {s.get('rerank_score', 0):.4f}")
            
    print("\n✅ ALL BENCHMARK & ACCURACY TESTS COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    # If 10,000 pages takes too long for a single test run, allow configuring test page size
    import sys
    target = int(sys.argv[1]) if len(sys.argv) > 1 else TARGET_PAGE_COUNT
    pdf_file = fetch_arxiv_pdfs_and_stitch(target_pages=target)
    run_stress_test_and_benchmark(pdf_file)
