"""
Markdown Table QA Pipeline with Map-Reduce Support
Pipeline for extracting tables from Markdown files and generating Q/A pairs using Map-Reduce approach
"""
import os
import glob
import pandas as pd

from dataflow.operators.knowledge_cleaning import (
    PDFTableQAGenerator,
    TableChunkSplitter,
    MapReduceQAGenerator,
)
from dataflow.utils.storage import FileStorage
from dataflow.serving import APILLMServing_request


class MarkdownTableQA_APIPipeline():
    """
    Pipeline for generating Q/A pairs from Markdown tables.
    
    Supports two modes:
    1. Direct mode: For small documents within token limits
    2. Map-Reduce mode: For large documents exceeding token limits
    
    Steps (Map-Reduce mode):
    1. Read Markdown files directly (no PDF conversion needed)
    2. Split document by table boundaries
    3. Generate summaries for each chunk (Map)
    4. Plan and generate Q/A pairs (Reduce)
    """
    
    # Default markdown directory
    DEFAULT_MARKDOWN_DIR = "/home/wangdeng/Github/dataflow/DataFlow/dataflow/example/markdown/Cleaned/Cleaned/01_Water/Data_Reports"
    
    def __init__(self, markdown_dir: str = None):
        """
        Initialize the pipeline.
        
        Args:
            markdown_dir: Directory containing markdown files to process.
                         If None, uses the default directory.
        """
        self.markdown_dir = markdown_dir or self.DEFAULT_MARKDOWN_DIR
        
        self.storage = FileStorage(
            first_entry_file_name=None,  # We'll create entries dynamically
            cache_path="./.cache/markdown_table_qa",
            file_name_prefix="markdown_table_qa_step",
            cache_type="json",
        )

        self.llm_serving = APILLMServing_request(
            api_url="https://oneapi.hkgai.net/v1/chat/completions",
            model_name="kimi-k2",
            max_workers=8,
            temperature=0.3,
        )

        # Step 1: Split by table boundaries
        self.table_chunk_splitter = TableChunkSplitter(
            max_chunk_size=20000,
            min_chunk_size=500,
            context_before=500,
            context_after=500,
        )

        # Step 2: Map-Reduce QA generation
        self.map_reduce_qa_generator = MapReduceQAGenerator(
            llm_serving=self.llm_serving,
            lang="en",
            max_qa=50,
            max_relevant_chunks=6,
        )

        # Legacy: Direct mode generator (for small documents)
        self.table_qa_generator_direct = PDFTableQAGenerator(
            llm_serving=self.llm_serving,
            lang="en",
            max_qa=50,
            max_text_length=1000000,
        )

    def _read_markdown_files(self) -> list:
        """
        Read all markdown files from the configured directory.
        
        Returns:
            List of dicts with 'source' (file path) and 'text' (file content)
        """
        markdown_files = glob.glob(os.path.join(self.markdown_dir, "*.md"))
        
        if not markdown_files:
            print(f"[Warning] No markdown files found in: {self.markdown_dir}")
            return []
        
        print(f"[Step 1] Found {len(markdown_files)} markdown files in {self.markdown_dir}")
        
        documents = []
        for file_path in sorted(markdown_files):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    documents.append({
                        "source": file_path,
                        "text": content,
                    })
                    print(f"  - Loaded: {os.path.basename(file_path)} ({len(content)} chars)")
            except Exception as e:
                print(f"  - Error reading {file_path}: {e}")
        
        return documents

    def forward(self, use_map_reduce: bool = True, merge_mode: bool = True):
        """
        Run the pipeline.
        
        Args:
            use_map_reduce: If True, use Map-Reduce approach for large documents.
                           If False, use direct approach (may fail for large docs).
            merge_mode: If True, merge all documents into one before generating Q/A.
                       This is useful when multiple documents contain related data.
        """
        # Step 1: Read Markdown files directly
        print("[Step 1] Reading Markdown files...")
        documents = self._read_markdown_files()
        
        if not documents:
            print("[Error] No documents to process!")
            return
        
        # Create cache directory
        os.makedirs("./.cache/markdown_table_qa", exist_ok=True)
        os.makedirs("./.cache/markdown_table_qa/tables", exist_ok=True)
        
        if merge_mode:
            # Merge all documents into one before processing
            print("[Merge Mode] Merging all documents...")
            
            all_texts = []
            all_sources = []
            for idx, doc in enumerate(documents):
                content = doc["text"]
                source = doc["source"]
                all_texts.append(f"\n\n--- Document {idx + 1}: {os.path.basename(source)} ---\n\n{content}")
                all_sources.append(source)
            
            # Merge all texts into one
            merged_text = "\n".join(all_texts)
            print(f"[Merge Mode] Merged {len(all_sources)} documents, total length: {len(merged_text)} chars")
            
            # Create merged dataframe
            merged_df = pd.DataFrame([{
                "source": " + ".join([os.path.basename(s) for s in all_sources]),
                "text": merged_text,
            }])
            
            if use_map_reduce:
                # Split merged text into chunks
                chunks = self.table_chunk_splitter.process(merged_text)
                print(f"[Merge Mode] Split into {len(chunks)} chunks")
                
                # Generate Q/A using Map-Reduce directly
                print("[Step 2] Generating Q/A pairs using Map-Reduce on merged documents...")
                csv_output_dir = "./.cache/markdown_table_qa/tables"
                result = self.map_reduce_qa_generator.process(chunks, csv_output_dir=csv_output_dir)
                
                merged_df['chunks'] = [chunks]
                merged_df['table_qa_pairs'] = [result['qa_pairs']]
                merged_df['chunk_summaries'] = [result['summaries']]
                merged_df['qa_plans'] = [result['plans']]
                merged_df['table_csv_paths'] = [result['csv_paths']]
                
                # Save results
                output_path = "./.cache/markdown_table_qa/markdown_table_qa_result.json"
                merged_df.to_json(output_path, orient='records', force_ascii=False, indent=2)
                
                print(f"[Merge Mode] Extracted {len(result['csv_paths'])} tables to CSV")
                print(f"[Merge Mode] Generated {len(result['qa_pairs'])} Q/A pairs from merged documents")
                
                # Also save QA pairs to a separate file for easy access
                qa_output_path = "./.cache/markdown_table_qa/qa_pairs.json"
                import json
                with open(qa_output_path, 'w', encoding='utf-8') as f:
                    json.dump(result['qa_pairs'], f, ensure_ascii=False, indent=2)
                print(f"[Merge Mode] QA pairs saved to: {qa_output_path}")
                
            else:
                # Direct processing on merged text
                print("[Step 2] Generating Q/A pairs directly on merged documents...")
                outputs = self.table_qa_generator_direct.process_batch([merged_text])
                
                # Extract tables to CSV
                csv_output_dir = "./.cache/markdown_table_qa/tables"
                csv_paths = self.table_qa_generator_direct._extract_tables_to_csv(
                    text=merged_text,
                    output_dir=csv_output_dir,
                    base_name="merged_docs"
                )
                
                result_df = pd.DataFrame([{
                    "source": " + ".join([os.path.basename(s) for s in all_sources]),
                    "text": merged_text,
                    "table_qa_pairs": outputs[0]['qa_pairs'] if outputs else [],
                    "has_table": outputs[0]['has_table'] if outputs else False,
                    "table_csv_paths": csv_paths,
                }])
                
                # Save results
                output_path = "./.cache/markdown_table_qa/markdown_table_qa_result.json"
                result_df.to_json(output_path, orient='records', force_ascii=False, indent=2)
            
            print("[Done] Merged Q/A generation complete!")
            print("Results saved to ./.cache/markdown_table_qa/")
        
        elif use_map_reduce:
            # Map-Reduce mode for large documents (process each separately)
            all_results = []
            
            for idx, doc in enumerate(documents):
                print(f"\n[Processing Document {idx + 1}/{len(documents)}] {os.path.basename(doc['source'])}")
                
                # Split document into chunks
                print("  [Step 1] Splitting document by table boundaries...")
                chunks = self.table_chunk_splitter.process(doc['text'])
                print(f"  [Step 1] Split into {len(chunks)} chunks")
                
                # Generate Q/A using Map-Reduce
                print("  [Step 2] Generating Q/A pairs using Map-Reduce...")
                csv_output_dir = "./.cache/markdown_table_qa/tables"
                result = self.map_reduce_qa_generator.process(chunks, csv_output_dir=csv_output_dir)
                
                all_results.append({
                    "source": doc['source'],
                    "chunks": chunks,
                    "table_qa_pairs": result['qa_pairs'],
                    "chunk_summaries": result['summaries'],
                    "qa_plans": result['plans'],
                    "table_csv_paths": result['csv_paths'],
                })
                
                print(f"  [Done] Generated {len(result['qa_pairs'])} Q/A pairs")
            
            # Save all results
            output_path = "./.cache/markdown_table_qa/markdown_table_qa_result.json"
            result_df = pd.DataFrame(all_results)
            result_df.to_json(output_path, orient='records', force_ascii=False, indent=2)
            
            print("\n[Done] Map-Reduce Q/A generation complete!")
            print("Results saved to ./.cache/markdown_table_qa/")
            
        else:
            # Direct mode for small documents
            all_results = []
            
            for idx, doc in enumerate(documents):
                print(f"\n[Processing Document {idx + 1}/{len(documents)}] {os.path.basename(doc['source'])}")
                
                print("  [Step 1] Generating Q/A pairs (direct mode)...")
                outputs = self.table_qa_generator_direct.process_batch([doc['text']])
                
                # Extract tables to CSV
                csv_output_dir = "./.cache/markdown_table_qa/tables"
                csv_paths = self.table_qa_generator_direct._extract_tables_to_csv(
                    text=doc['text'],
                    output_dir=csv_output_dir,
                    base_name=os.path.basename(doc['source']).replace('.md', '')
                )
                
                all_results.append({
                    "source": doc['source'],
                    "text": doc['text'],
                    "table_qa_pairs": outputs[0]['qa_pairs'] if outputs else [],
                    "has_table": outputs[0]['has_table'] if outputs else False,
                    "table_csv_paths": csv_paths,
                })
                
                qa_count = len(outputs[0]['qa_pairs']) if outputs else 0
                print(f"  [Done] Generated {qa_count} Q/A pairs")
            
            # Save all results
            output_path = "./.cache/markdown_table_qa/markdown_table_qa_result.json"
            result_df = pd.DataFrame(all_results)
            result_df.to_json(output_path, orient='records', force_ascii=False, indent=2)
            
            print("\n[Done] Direct Q/A generation complete!")
            print("Results saved to ./.cache/markdown_table_qa/")


if __name__ == "__main__":
    pipeline = MarkdownTableQA_APIPipeline()
    # use_map_reduce=True: For large documents (recommended)
    # use_map_reduce=False: For small documents within token limits
    # merge_mode=True: Merge all documents before generating Q/A
    # merge_mode=False: Process each document separately
    pipeline.forward(use_map_reduce=True, merge_mode=True)
