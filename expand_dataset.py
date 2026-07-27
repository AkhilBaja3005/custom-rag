import fitz
import sys

def main():
    base_file = "arxiv_10000_pages_master.pdf"
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    
    doc = fitz.open(base_file)
    initial_pages = len(doc)
    print(f"Base seed PDF has {initial_pages} pages. Expanding to {target} pages...")
    
    while len(doc) < target:
        doc.insert_pdf(fitz.open(base_file))
        print(f"  + Expanded PDF size: {len(doc)} / {target} pages...")
        
    # Trim to exact target page count
    if len(doc) > target:
        doc.select(list(range(target)))
        
    print(f"Saving final {len(doc)}-page master dataset to '{base_file}'...")
    doc.save("arxiv_10000_pages_master_full.pdf")
    doc.close()
    
    import os
    os.replace("arxiv_10000_pages_master_full.pdf", base_file)
    print("Done! Master 10,000-page dataset ready.")

if __name__ == "__main__":
    main()
