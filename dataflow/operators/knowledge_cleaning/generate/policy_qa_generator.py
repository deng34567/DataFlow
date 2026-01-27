"""
Policy QA Generator Operator
Generates Q/A pairs from policy/regulation documents using Map-Reduce approach.
Unlike MapReduceQAGenerator, this focuses on text-based content rather than tables.
"""
import json
import re
import pandas as pd
from typing import Any, Dict, List, Optional, Union
from tqdm import tqdm

import logging
logging.getLogger('dataflow').setLevel(logging.WARNING)

from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC
from dataflow.prompts.policy_qa_prompt import PolicySummaryPrompt, PolicyQAPlannerPrompt, PolicyQAPrompt
from dataflow.core.prompt import prompt_restrict, DIYPromptABC


class PolicyQAGenerator(OperatorABC):
    """
    Generate Q/A pairs from policy/regulation documents using Map-Reduce approach.
    
    Map phase: Generate summaries for each chunk (extracting policy content)
    Reduce phase: Plan and generate policy-focused Q/A pairs
    
    Unlike MapReduceQAGenerator, this does NOT filter by has_table flag,
    as policy documents typically contain text-based content.
    """
    
    def __init__(
        self,
        llm_serving: LLMServingABC,
        lang: str = "en",
        max_qa: int = 50,
        max_relevant_chunks: int = 5,
    ):
        """
        Initialize the PolicyQAGenerator.
        
        Args:
            llm_serving: LLM serving instance
            lang: Language for prompts ("en" or "zh")
            max_qa: Maximum number of Q/A pairs to generate
            max_relevant_chunks: Max chunks to include per question
        """
        self.llm_serving = llm_serving
        self.lang = lang
        self.max_qa = max_qa
        self.max_relevant_chunks = max_relevant_chunks
        
        self.logger = logging.getLogger('dataflow.operators.policy_qa_generator')
        
        # Initialize policy-specific prompts
        self.summary_prompt = PolicySummaryPrompt(lang=lang)
        self.planner_prompt = PolicyQAPlannerPrompt(lang=lang)
        self.qa_prompt = PolicyQAPrompt(lang=lang)
        
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return {
                "name": "政策问答生成器",
                "desc": "使用Map-Reduce方法从政策法规文档生成问答对",
            }
        else:
            return {
                "name": "PolicyQAGenerator",
                "desc": "Generates Q/A pairs from policy documents using Map-Reduce approach",
            }

    def _generate_chunk_summaries(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Map phase: Generate summaries for each chunk."""
        self.logger.info(f"Generating summaries for {len(chunks)} chunks...")
        
        summaries = []
        
        # Process all chunks (no filtering by has_table)
        if chunks:
            user_inputs = []
            for chunk in chunks:
                prompt = self.summary_prompt.build_prompt(chunk['id'], chunk['text'])
                user_inputs.append(prompt)
            
            sys_prompt = self.summary_prompt.build_system_prompt()
            responses = self.llm_serving.generate_from_input(
                user_inputs=user_inputs,
                system_prompt=sys_prompt
            )
            
            for chunk, response in zip(chunks, responses):
                summary = self._parse_summary_response(response, chunk['id'])
                summaries.append(summary)
        
        # Sort by chunk_id
        summaries.sort(key=lambda x: x.get('chunk_id', 0))
        return summaries

    def _parse_summary_response(self, response: str, chunk_id: int) -> Dict[str, Any]:
        """Parse LLM response for chunk summary."""
        try:
            # Clean response and extract JSON
            cleaned = self._clean_llm_response(response)
            
            # Try to parse JSON
            summary = json.loads(cleaned)
            summary['chunk_id'] = chunk_id
            
            # Determine if this chunk has meaningful content
            has_content = (
                bool(summary.get('articles')) or
                bool(summary.get('definitions')) or
                bool(summary.get('responsibilities')) or
                bool(summary.get('penalties')) or
                bool(summary.get('key_provisions'))
            )
            summary['has_content'] = has_content
            
            return summary
            
        except json.JSONDecodeError as e:
            self.logger.warning(f"Failed to parse summary for chunk {chunk_id}: {e}")
            return {
                'chunk_id': chunk_id,
                'regulation_info': {},
                'articles': [],
                'definitions': [],
                'scope': '',
                'responsibilities': [],
                'penalties': [],
                'key_provisions': [],
                'has_content': False,
                'parse_error': str(e)
            }

    def _plan_qa_questions(self, summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Plan which questions to generate based on summaries."""
        self.logger.info("Planning Q/A questions based on summaries...")
        
        # Filter summaries with meaningful content
        content_summaries = [s for s in summaries if s.get('has_content', False)]
        
        if not content_summaries:
            self.logger.warning("No summaries with meaningful content found")
            # Fall back to using all summaries if no content detected
            content_summaries = summaries
        
        # Create condensed summary text for planning
        summary_text = json.dumps(content_summaries, ensure_ascii=False, indent=2)
        
        # Check if summary is too long
        # 504 Gateway Timeout happens often with large prompts
        if len(summary_text) > 15000:
            self.logger.info("Summary text too long, truncating for planning...")
            # Truncate each summary aggressively
            truncated = []
            for s in content_summaries:
                # Create a minimal summary for planning
                minimal_s = {
                    'chunk_id': s.get('chunk_id'),
                    'regulation_info': s.get('regulation_info'),
                }
                
                # Add limited articles (titles and very brief summary)
                if s.get('articles'):
                    minimal_s['articles'] = []
                    for art in s.get('articles', [])[:5]:
                        minimal_s['articles'].append({
                            'article_number': art.get('article_number'),
                            'title': art.get('title'),
                            # Skip full content summary if it's long
                        })
                
                # Add limited definitions (names only)
                if s.get('definitions'):
                    minimal_s['definitions'] = [d.get('term') for d in s.get('definitions', [])[:5]]
                    
                truncated.append(minimal_s)
                
            summary_text = json.dumps(truncated, ensure_ascii=False, indent=2)
        
        prompt = self.planner_prompt.build_prompt(summary_text, self.max_qa)
        sys_prompt = self.planner_prompt.build_system_prompt()
        
        responses = self.llm_serving.generate_from_input(
            user_inputs=[prompt],
            system_prompt=sys_prompt
        )
        
        return self._parse_qa_plan(responses[0]) if responses else []

    def _clean_llm_response(self, response: str) -> str:
        """
        Clean LLM response by removing thinking tags and extracting content from code blocks.
        
        Handles:
        1. <think>...</think> tags (Qwen models thinking chain)
        2. ```json ... ``` markdown code blocks
        """
        if not response:
            return ""
        
        # Remove <think>...</think> blocks first
        cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        
        # Try to extract from markdown code blocks
        code_block_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', cleaned, re.DOTALL)
        if code_block_match:
            cleaned = code_block_match.group(1)
        
        # Strip whitespace
        cleaned = cleaned.strip()
        
        return cleaned

    def _parse_qa_plan(self, response: str) -> List[Dict[str, Any]]:
        """Parse QA plan from LLM response."""
        try:
            cleaned = self._clean_llm_response(response)
            plans = json.loads(cleaned)
            
            if isinstance(plans, list):
                valid_plans = []
                for plan in plans:
                    if isinstance(plan, dict) and 'description' in plan:
                        valid_plans.append(plan)
                return valid_plans
            elif isinstance(plans, dict):
                return [plans] if 'description' in plans else []
            
            return []
            
        except json.JSONDecodeError as e:
            self.logger.warning(f"Failed to parse QA plan: {e}")
            # Try line-by-line parsing
            try:
                lines = response.strip().split('\n')
                for line in lines:
                    line = line.strip()
                    if line.startswith('[') or line.startswith('{'):
                        cleaned = self._clean_llm_response(line)
                        plans = json.loads(cleaned)
                        if isinstance(plans, list):
                            return plans
            except:
                pass
            return []

    def _generate_qa_from_plan(
        self,
        qa_plans: List[Dict[str, Any]],
        chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Reduce phase: Generate Q/A pairs from plans."""
        self.logger.info(f"Generating Q/A pairs from {len(qa_plans)} plans...")
        
        # Create chunk lookup
        chunk_map = {c['id']: c for c in chunks}
        
        user_inputs = []
        valid_plans = []
        
        for plan in qa_plans:
            # Get relevant chunks
            chunk_ids = plan.get('required_chunk_ids', [])
            if not chunk_ids:
                # If no specific chunks, use all chunks
                chunk_ids = list(chunk_map.keys())[:self.max_relevant_chunks]
            
            # Limit chunks
            chunk_ids = chunk_ids[:self.max_relevant_chunks]
            
            relevant_chunks = []
            for cid in chunk_ids:
                if cid in chunk_map:
                    relevant_chunks.append({
                        'id': cid,
                        'text': chunk_map[cid]['text'][:8000]  # Limit chunk size
                    })
            
            if not relevant_chunks:
                continue
            
            chunks_text = json.dumps(relevant_chunks, ensure_ascii=False, indent=2)
            plan_text = json.dumps(plan, ensure_ascii=False, indent=2)
            
            prompt = self.qa_prompt.build_prompt(plan_text, chunks_text)
            user_inputs.append(prompt)
            valid_plans.append(plan)
        
        if not user_inputs:
            return []
        
        sys_prompt = self.qa_prompt.build_system_prompt()
        responses = self.llm_serving.generate_from_input(
            user_inputs=user_inputs,
            system_prompt=sys_prompt
        )
        
        qa_pairs = []
        for plan, response in zip(valid_plans, responses):
            qa = self._parse_qa_response(response)
            if qa:
                qa['plan'] = plan
                qa_pairs.append(qa)
        
        return qa_pairs

    def _parse_qa_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse Q/A from LLM response."""
        try:
            cleaned = self._clean_llm_response(response)
            qa = json.loads(cleaned)
            
            if isinstance(qa, dict) and 'question' in qa and 'answer' in qa:
                return qa
            
            return None
            
        except json.JSONDecodeError as e:
            self.logger.warning(f"Failed to parse QA response: {e}")
            return None

    def process(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process chunks using Map-Reduce approach.
        
        Args:
            chunks: List of document chunks with 'id' and 'text' fields
            
        Returns:
            Dictionary with qa_pairs, summaries, and plans
        """
        if not chunks:
            return {
                'qa_pairs': [],
                'summaries': [],
                'plans': []
            }
        
        # Map phase: Generate summaries
        print(f"  [Map Phase] Generating summaries for {len(chunks)} chunks...")
        summaries = self._generate_chunk_summaries(chunks)
        print(f"  [Map Phase] Generated {len(summaries)} summaries")
        
        # Count chunks with content
        content_count = sum(1 for s in summaries if s.get('has_content', False))
        print(f"  [Map Phase] {content_count} chunks have meaningful policy content")
        
        # Plan phase
        print(f"  [Plan Phase] Planning Q/A questions...")
        qa_plans = self._plan_qa_questions(summaries)
        print(f"  [Plan Phase] Planned {len(qa_plans)} questions")
        
        # Reduce phase: Generate Q/A
        print(f"  [Reduce Phase] Generating Q/A pairs...")
        qa_pairs = self._generate_qa_from_plan(qa_plans, chunks)
        print(f"  [Reduce Phase] Generated {len(qa_pairs)} Q/A pairs")
        
        return {
            'qa_pairs': qa_pairs,
            'summaries': summaries,
            'plans': qa_plans
        }

    def run(
        self,
        storage = None,
        input_key: str = 'chunks',
        output_key: str = 'policy_qa_pairs',
    ):
        """
        Run the Policy QA Generator.
        
        Args:
            storage: DataFlowStorage instance
            input_key: Key for input chunks
            output_key: Key for output QA pairs
        """
        if storage is None:
            raise ValueError("Storage is required")
        
        # Get chunks from storage
        chunks = storage.get(input_key)
        if chunks is None:
            self.logger.warning(f"No data found for key '{input_key}'")
            return
        
        # Process
        result = self.process(chunks)
        
        # Store results
        storage.set(output_key, result['qa_pairs'])
        storage.set(f"{output_key}_summaries", result['summaries'])
        storage.set(f"{output_key}_plans", result['plans'])
        
        self.logger.info(f"Generated {len(result['qa_pairs'])} Q/A pairs")
