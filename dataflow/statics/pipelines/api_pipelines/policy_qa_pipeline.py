"""
Policy QA Pipeline with Map-Reduce Support
Pipeline for generating Q/A pairs from policy/regulation Markdown files.

Unlike pdf_table_qa_pipeline(markdown).py which focuses on table-based data,
this pipeline is designed for text-based policy content (laws, regulations, standards).
"""
import os
import glob
import pandas as pd

from dataflow.operators.knowledge_cleaning import (
    TableChunkSplitter,
)
from dataflow.operators.knowledge_cleaning.generate.policy_qa_generator import PolicyQAGenerator
from dataflow.utils.storage import FileStorage
from dataflow.serving import APILLMServing_request


class PolicyQA_APIPipeline():
    """
    Pipeline for generating Q/A pairs from policy/regulation Markdown files.
    
    Uses Map-Reduce approach:
    1. Read Markdown files containing policy/regulation content
    2. Split document into manageable chunks
    3. Generate summaries for each chunk (Map) - extracting policy elements
    4. Plan and generate Q/A pairs (Reduce) - text-based policy questions
    """
    
    # Default markdown directory for policy documents
    DEFAULT_MARKDOWN_DIR = "/home/wangdeng/Github/dataflow/DataFlow/dataflow/example/markdown/Cleaned/Cleaned/02_Atmosphere/Laws_Regulations"
    
    def __init__(self, markdown_dir: str = None):
        """
        Initialize the pipeline.
        
        Args:
            markdown_dir: Directory containing markdown files to process.
                         If None, uses the default policy documents directory.
        """
        self.markdown_dir = markdown_dir or self.DEFAULT_MARKDOWN_DIR
        
        self.storage = FileStorage(
            first_entry_file_name=None,
            cache_path="./.cache/policy_qa",
            file_name_prefix="policy_qa_step",
            cache_type="json",
        )

        self.llm_serving = APILLMServing_request(
            api_url="https://oneapi.hkgai.net/v1/chat/completions",
            model_name="kimi-k2",
            max_workers=4,  # Reduced to avoid rate limiting
            temperature=0.3,
            timeout=(10.0, 300.0),  # Increased read timeout for policy processing
        )

        # Step 1: Split into chunks (reuse existing splitter)
        self.chunk_splitter = TableChunkSplitter(
            max_chunk_size=8000,  # Smaller chunks for policy text
            min_chunk_size=500,
            context_before=300,
            context_after=300,
        )

        # Step 2: Policy QA generation using Map-Reduce
        self.policy_qa_generator = PolicyQAGenerator(
            llm_serving=self.llm_serving,
            lang="zh",
            max_qa=100,
            max_relevant_chunks=5,
        )

    def _get_output_prefix(self) -> str:
        """
        Generate output file prefix based on markdown directory path.
        This ensures outputs from different directories don't overwrite each other.
        
        Returns:
            A sanitized string suitable for use in filenames
        """
        import re
        # Get the last 2 parts of the path for a meaningful prefix
        path_parts = self.markdown_dir.rstrip('/').split('/')
        if len(path_parts) >= 2:
            prefix = f"{path_parts[-2]}_{path_parts[-1]}"
        else:
            prefix = path_parts[-1] if path_parts else "default"
        
        # Sanitize: replace invalid filename characters with underscores
        prefix = re.sub(r'[^\w\-]', '_', prefix)
        return prefix

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

    def forward(self, merge_mode: bool = True, max_files: int = None):
        """
        Run the pipeline.
        
        Args:
            merge_mode: If True, merge all documents into one before generating Q/A.
                       This is useful when multiple documents contain related content.
            max_files: If set, limit the number of files to process (for testing)
        """
        # Step 1: Read Markdown files
        print("=" * 60)
        print("[Policy QA Pipeline] Starting...")
        print("=" * 60)
        print(f"\n[Step 1] Reading Markdown files from: {self.markdown_dir}")
        documents = self._read_markdown_files()
        
        if not documents:
            print("[Error] No documents to process!")
            return
        
        if max_files:
            documents = documents[:max_files]
            print(f"[Info] Limited to {max_files} files for processing")
        
        # Get output prefix based on markdown directory
        output_prefix = self._get_output_prefix()
        print(f"[Info] Output prefix: {output_prefix}")
        
        # Create cache directory
        cache_dir = f"./.cache/policy_qa/{output_prefix}"
        os.makedirs(cache_dir, exist_ok=True)
        
        if merge_mode:
            # Merge all documents into one before processing
            print("\n[Merge Mode] Merging all documents...")
            
            all_texts = []
            all_sources = []
            for idx, doc in enumerate(documents):
                content = doc["text"]
                source = doc["source"]
                all_texts.append(f"\n\n--- 文档 {idx + 1}: {os.path.basename(source)} ---\n\n{content}")
                all_sources.append(source)
            
            # Merge all texts into one
            merged_text = "\n".join(all_texts)
            print(f"[Merge Mode] Merged {len(all_sources)} documents, total length: {len(merged_text)} chars")
            
            # Split merged text into chunks
            print("\n[Step 2] Splitting merged document into chunks...")
            chunks = self.chunk_splitter.process(merged_text)
            print(f"[Step 2] Split into {len(chunks)} chunks")
            
            # Generate Q/A using Map-Reduce
            print("\n[Step 3] Generating Q/A pairs using Map-Reduce...")
            result = self.policy_qa_generator.process(chunks)
            
            # Create result dataframe
            merged_df = pd.DataFrame([{
                "source": " + ".join([os.path.basename(s) for s in all_sources]),
                "text_length": len(merged_text),
                "num_chunks": len(chunks),
                "chunks": chunks,
                "policy_qa_pairs": result['qa_pairs'],
                "chunk_summaries": result['summaries'],
                "qa_plans": result['plans'],
            }])
            
            # Save results
            output_path = f"{cache_dir}/{output_prefix}_policy_qa_result.json"
            merged_df.to_json(output_path, orient='records', force_ascii=False, indent=2)
            
            print(f"\n[Merge Mode] Generated {len(result['qa_pairs'])} Q/A pairs from merged documents")
            
            # Also save QA pairs to a separate file for easy access
            qa_output_path = f"{cache_dir}/{output_prefix}_qa_pairs.json"
            import json
            with open(qa_output_path, 'w', encoding='utf-8') as f:
                json.dump(result['qa_pairs'], f, ensure_ascii=False, indent=2)
            print(f"[Merge Mode] QA pairs saved to: {qa_output_path}")
            
        else:
            # Process each document separately
            all_results = []
            
            for idx, doc in enumerate(documents):
                print(f"\n[Processing Document {idx + 1}/{len(documents)}] {os.path.basename(doc['source'])}")
                
                # Split document into chunks
                print("  [Step 1] Splitting document into chunks...")
                chunks = self.chunk_splitter.process(doc['text'])
                print(f"  [Step 1] Split into {len(chunks)} chunks")
                
                # Generate Q/A using Map-Reduce
                print("  [Step 2] Generating Q/A pairs...")
                result = self.policy_qa_generator.process(chunks)
                
                all_results.append({
                    "source": doc['source'],
                    "text_length": len(doc['text']),
                    "num_chunks": len(chunks),
                    "chunks": chunks,
                    "policy_qa_pairs": result['qa_pairs'],
                    "chunk_summaries": result['summaries'],
                    "qa_plans": result['plans'],
                })
                
                print(f"  [Done] Generated {len(result['qa_pairs'])} Q/A pairs")
            
            # Save all results
            output_path = f"{cache_dir}/{output_prefix}_policy_qa_result.json"
            result_df = pd.DataFrame(all_results)
            result_df.to_json(output_path, orient='records', force_ascii=False, indent=2)
            
            print(f"\n[Done] Generated Q/A pairs for {len(documents)} documents")
            print(f"Results saved to {cache_dir}/")
        
        print("\n" + "=" * 60)
        print("[Policy QA Pipeline] Complete!")
        print("=" * 60)


if __name__ == "__main__":
    # ============================================================
    # 使用说明：
    # 1. 如需更换 markdown 文件夹，直接修改下面的 markdown_dir 参数
    # 2. 输出文件会自动包含文件夹名称，不会覆盖之前的结果
    # 3. 首次测试可以设置 max_files 限制处理的文件数量
    # ============================================================
    
    # 方式1：使用默认目录 (Laws_Regulations)
    pipeline = PolicyQA_APIPipeline()
    
    # 方式2：指定自定义目录
    # markdown_dir = "/path/to/your/markdown/files"
    # pipeline = PolicyQA_APIPipeline(markdown_dir=markdown_dir)
    
    # 运行 Pipeline
    # merge_mode=True: 合并所有文档后生成 Q/A（推荐）
    # merge_mode=False: 对每个文档单独生成 Q/A
    # max_files=5: 限制处理的文件数量（用于测试）
    pipeline.forward(merge_mode=True, max_files=None)
