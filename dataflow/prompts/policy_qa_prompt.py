"""
Prompt templates for policy/regulation Q/A generation.
Designed for text-based policy documents (laws, regulations, standards).
"""
import textwrap
from dataflow.utils.registry import PROMPT_REGISTRY
from dataflow.core.prompt import PromptABC


@PROMPT_REGISTRY.register()
class PolicySummaryPrompt(PromptABC):
    """
    Prompt for extracting structured summaries from policy/regulation documents.
    Used in the Map phase of Map-Reduce QA generation for policy content.
    """
    def __init__(self, lang: str = "en"):
        self.lang = lang
        self.system_text = self.build_system_prompt()
        
    def build_system_prompt(self) -> str:
        if self.lang == "en":
            return textwrap.dedent("""\
                You are a legal/policy document analysis expert. Your task is to extract structured information from policy and regulation document chunks.
                
                For each chunk, you must identify and extract:
                1. Regulation name or title (if present)
                2. Key articles/clauses and their main content
                3. Definitions of important terms
                4. Scope of application (who/what it applies to)
                5. Responsibilities and obligations
                6. Penalties and consequences for violations
                7. Implementation dates or deadlines
                8. References to other laws or regulations
                
                Output ONLY valid JSON with this structure:
                {
                    "chunk_id": <provided id>,
                    "regulation_info": {
                        "name": "Name of the regulation if mentioned",
                        "issuing_authority": "Authority that issued the regulation",
                        "effective_date": "When it takes effect"
                    },
                    "articles": [
                        {
                            "article_number": "Article/Chapter number",
                            "title": "Article title if any",
                            "content_summary": "Brief summary of article content"
                        }
                    ],
                    "definitions": [
                        {"term": "Term name", "definition": "Definition text"}
                    ],
                    "scope": "Description of who/what this applies to",
                    "responsibilities": [
                        {"entity": "Responsible party", "obligation": "What they must do"}
                    ],
                    "penalties": [
                        {"violation": "Type of violation", "consequence": "Penalty description"}
                    ],
                    "key_provisions": [
                        "Important provisions or requirements mentioned"
                    ]
                }
                """)
        else:
            return textwrap.dedent("""\
                您是法律/政策文件分析专家。您的任务是从政策法规文档片段中提取结构化信息。
                
                对于每个片段，您必须识别和提取：
                1. 法规名称或标题（如有）
                2. 关键条款及其主要内容
                3. 重要术语的定义
                4. 适用范围（适用于谁/什么）
                5. 责任和义务
                6. 违规处罚和后果
                7. 实施日期或截止日期
                8. 对其他法律法规的引用
                
                仅输出以下结构的有效JSON：
                {
                    "chunk_id": <提供的id>,
                    "regulation_info": {
                        "name": "法规名称（如提及）",
                        "issuing_authority": "发布法规的机关",
                        "effective_date": "生效时间"
                    },
                    "articles": [
                        {
                            "article_number": "条/章编号",
                            "title": "条款标题（如有）",
                            "content_summary": "条款内容简要概述"
                        }
                    ],
                    "definitions": [
                        {"term": "术语名称", "definition": "定义内容"}
                    ],
                    "scope": "适用范围描述",
                    "responsibilities": [
                        {"entity": "责任主体", "obligation": "应履行的义务"}
                    ],
                    "penalties": [
                        {"violation": "违规类型", "consequence": "处罚描述"}
                    ],
                    "key_provisions": [
                        "提及的重要规定或要求"
                    ]
                }
                """)

    def build_prompt(self, chunk_id: int, chunk_text: str) -> str:
        if self.lang == "en":
            return textwrap.dedent(f"""\
                Analyze the following policy/regulation document chunk (ID: {chunk_id}) and extract structured information.
                
                Document Chunk:
                {chunk_text}
                
                Extract all regulation info, articles, definitions, scope, responsibilities, penalties, and key provisions. Output valid JSON only.
                """)
        else:
            return textwrap.dedent(f"""\
                分析以下政策法规文档片段（ID: {chunk_id}）并提取结构化信息。
                
                文档片段：
                {chunk_text}
                
                提取所有法规信息、条款、定义、适用范围、责任、处罚和关键规定。仅输出有效的JSON。
                """)


@PROMPT_REGISTRY.register()
class PolicyQAPlannerPrompt(PromptABC):
    """
    Prompt for planning Q/A generation from policy documents.
    """
    def __init__(self, lang: str = "en"):
        self.lang = lang
        self.system_text = self.build_system_prompt()
        
    def build_system_prompt(self) -> str:
        if self.lang == "en":
            return textwrap.dedent("""\
                You are a policy Q/A planning expert. Given summaries of policy document chunks, you need to:
                1. Identify important concepts that deserve explanation questions
                2. Find regulatory requirements that can form procedural questions
                3. Locate responsibility assignments that can form accountability questions
                4. Identify penalty clauses for violation-consequence questions
                5. Plan comparison questions between different provisions
                
                Focus on questions that:
                - Explain key legal/policy concepts
                - Clarify who is responsible for what
                - Describe required procedures or processes
                - Explain consequences of non-compliance
                - Compare or contrast different regulatory requirements
                
                Output ONLY valid JSON array with this structure:
                [
                    {
                        "question_type": "definition|procedure|responsibility|penalty|comparison|application",
                        "description": "Brief description of the question to generate",
                        "required_chunk_ids": [1, 3, 5],
                        "target_content": "Specific content to ask about",
                        "difficulty": "basic|intermediate|advanced"
                    }
                ]
                """)
        else:
            return textwrap.dedent("""\
                您是政策问答规划专家。根据政策文档片段的摘要，您需要：
                1. 识别需要解释的重要概念
                2. 找到可以形成程序性问题的监管要求
                3. 找到可以形成责任问题的责任分配
                4. 识别违规处罚条款
                5. 规划不同规定之间的比较问题
                
                关注以下类型的问题：
                - 解释关键的法律/政策概念
                - 明确谁负责什么
                - 描述所需的程序或流程
                - 解释违规的后果
                - 比较或对比不同的监管要求
                
                仅输出以下结构的有效JSON数组：
                [
                    {
                        "question_type": "定义|程序|责任|处罚|比较|适用",
                        "description": "要生成问题的简要描述",
                        "required_chunk_ids": [1, 3, 5],
                        "target_content": "要询问的具体内容",
                        "difficulty": "基础|中级|高级"
                    }
                ]
                """)

    def build_prompt(self, summaries: str, max_questions: int = 50) -> str:
        if self.lang == "en":
            return textwrap.dedent(f"""\
                Based on the following policy document summaries, plan questions (UP TO {max_questions} maximum).
                
                IMPORTANT: Generate as many meaningful questions as the content supports, but no more than {max_questions}.
                - If the content only supports 5 good questions, generate 5.
                - If the content is rich with policy details, generate more (up to {max_questions}).
                - Focus on quality over quantity. Prioritize questions that test understanding of the regulations.
                
                Question types to consider:
                - Definition questions: What is X? How is X defined?
                - Procedure questions: What steps must be followed to do X?
                - Responsibility questions: Who is responsible for X? What must entity Y do?
                - Penalty questions: What are the consequences of violating X?
                - Application questions: When does regulation X apply? Who must comply with X?
                - Comparison questions: How does X differ from Y?
                
                Chunk Summaries:
                {summaries}
                
                Output a JSON array of question plans. Generate only as many as the content meaningfully supports.
                """)
        else:
            return textwrap.dedent(f"""\
                根据以下政策文档摘要，规划问题（最多{max_questions}个）。
                
                重要：根据内容支持程度生成尽可能多的有意义问题，但不超过{max_questions}个。
                - 如果内容只能支持5个好问题，就生成5个。
                - 如果内容政策细节丰富，可以生成更多（最多{max_questions}个）。
                - 注重质量而非数量。优先考虑测试对法规理解的问题。
                
                需要考虑的问题类型：
                - 定义类问题：什么是X？X是如何定义的？
                - 程序类问题：做X需要遵循哪些步骤？
                - 责任类问题：谁负责X？Y主体必须做什么？
                - 处罚类问题：违反X的后果是什么？
                - 适用类问题：法规X何时适用？谁必须遵守X？
                - 比较类问题：X与Y有何不同？
                
                片段摘要：
                {summaries}
                
                输出问题计划的JSON数组。只生成内容能够有意义支持的数量。
                """)


@PROMPT_REGISTRY.register()
class PolicyQAPrompt(PromptABC):
    """
    Prompt for generating Q/A from policy documents based on QA plan.
    Unlike TargetedQAPrompt, this generates text-based answers instead of numerical values.
    """
    def __init__(self, lang: str = "en"):
        self.lang = lang
        self.system_text = self.build_system_prompt()
        
    def build_system_prompt(self) -> str:
        if self.lang == "en":
            return textwrap.dedent("""\
                You are a policy Q/A generation expert. Generate a question and answer based on:
                1. The question plan provided
                2. The relevant document chunks
                
                The question should:
                - Be clear and specific about what is being asked
                - Focus on practical understanding of the policy/regulation
                - Be answerable from the provided document content
                
                The answer should:
                - Be comprehensive but concise
                - Cite specific articles or provisions when relevant
                - Explain the regulation in plain language
                - Include any important conditions or exceptions
                
                Output ONLY valid JSON:
                {
                    "question": "The question text",
                    "answer": "The comprehensive answer based on the policy content",
                    "source_articles": ["Article X", "Article Y"],
                    "question_type": "definition|procedure|responsibility|penalty|comparison|application",
                    "key_terms": ["term1", "term2"]
                }
                """)
        else:
            return textwrap.dedent("""\
                您是政策问答生成专家。根据以下内容生成问答：
                1. 提供的问题计划
                2. 相关的文档片段
                
                问题应该：
                - 清晰明确地说明要问什么
                - 关注对政策/法规的实际理解
                - 可以从提供的文档内容中回答
                
                答案应该：
                - 全面但简洁
                - 在相关时引用具体条款或规定
                - 用通俗语言解释法规
                - 包含任何重要的条件或例外情况
                
                仅输出有效的JSON：
                {
                    "question": "问题文本",
                    "answer": "基于政策内容的全面答案",
                    "source_articles": ["第X条", "第Y条"],
                    "question_type": "定义|程序|责任|处罚|比较|适用",
                    "key_terms": ["术语1", "术语2"]
                }
                """)

    def build_prompt(self, question_plan: str, chunks: str) -> str:
        if self.lang == "en":
            return textwrap.dedent(f"""\
                Generate a policy Q/A pair based on the following plan and document chunks.
                
                Question Plan:
                {question_plan}
                
                Relevant Document Chunks:
                {chunks}
                
                Generate a clear question and comprehensive answer. The answer should be based on the actual policy content provided. Output valid JSON only.
                """)
        else:
            return textwrap.dedent(f"""\
                根据以下计划和文档片段生成政策问答对。
                
                问题计划：
                {question_plan}
                
                相关文档片段：
                {chunks}
                
                生成清晰的问题和全面的答案。答案应基于提供的实际政策内容。仅输出有效的JSON。
                """)
