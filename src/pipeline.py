from __future__ import annotations

import os
import time

from dataclasses import dataclass
from typing import Any

from pdf_loader import load_pdf
from preprocessing import preprocess_document
from chunking import chunk_document
from retrieval import retrieve_all
from prompt_builder import build_prompt_from_comparison
from llm import generate_answer

# ============================================================
# Pipeline Result
# ============================================================

@dataclass
class PipelineResult:
    """
    Final object returned by SmartDocPipeline.ask().

    Instead of returning only the LLM answer,
    we preserve everything generated during the pipeline.

    This allows:

        • Streamlit UI
        • Visualizer
        • Debugging
        • Educational explanation

    all to reuse the same object.
    """

    question: str

    answer: dict[str, Any]

    retrieval: Any

    prompt_package: Any

    timings: dict[str, float]

    trace: dict[str, Any]
    class SmartDocPipeline:

     def __init__(self) -> None:
        """
        Create an empty pipeline.

        Nothing is loaded here.

        The expensive work happens only once inside
        index_document().
        """

        self.document = None

        self.chunks = None

        self.index_ready = False

        self.document_name = ""

        self.statistics = {}

        self.trace = {}

    # --------------------------------------------------------
    # Reset
    # --------------------------------------------------------

    def reset(self) -> None:
        """
        Remove every cached object.

        Useful when a different PDF is uploaded.

        Example
        -------

            pipeline.reset()

            pipeline.index_document(new_pdf)
        """

        self.document = None

        self.chunks = None

        self.index_ready = False

        self.document_name = ""

        self.statistics = {}

        self.trace = {}

        print("[pipeline] Pipeline reset.")

            # --------------------------------------------------------
    # Index Document
    # --------------------------------------------------------

    def index_document(self, pdf_path: str) -> dict:
        """
        Build the complete document index.

        Pipeline:

            PDF
                ↓
            Load
                ↓
            Preprocess
                ↓
            Chunk
                ↓
            Ready for Retrieval

        This function performs every expensive preprocessing step
        only once.

        Parameters
        ----------
        pdf_path : str

            Path to the PDF.

        Returns
        -------
        dict

            Summary statistics describing the indexed document.
        """

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(
                f"PDF not found:\n{pdf_path}"
            )

        self.reset()

        self.document_name = os.path.basename(pdf_path)

        print("\n" + "=" * 70)
        print(" SmartDocAI — Building Document Index")
        print("=" * 70)

        total_start = time.perf_counter()

        # ----------------------------------------------------
        # Step 1
        # ----------------------------------------------------

        print("\nStep 1/3 : Loading PDF...")

        start = time.perf_counter()

        self.document = load_pdf(pdf_path)

        pdf_time = round(
            time.perf_counter() - start,
            3,
        )

        self.trace["pdf_loader"] = {

            "pages": len(self.document.pages),

            "time": pdf_time,

        }

        # ----------------------------------------------------
        # Step 2
        # ----------------------------------------------------

        print("\nStep 2/3 : Preprocessing...")

        start = time.perf_counter()

        preprocess_document(self.document)

        preprocessing_time = round(

            time.perf_counter() - start,

            3,

        )

        self.trace["preprocessing"] = {

            "time": preprocessing_time,

        }

        # ----------------------------------------------------
        # Step 3
        # ----------------------------------------------------

        print("\nStep 3/3 : Chunking...")

        start = time.perf_counter()

        self.chunks = chunk_document(

            self.document,

            target_sentences=10,

            overlap_sentences=2,

        )

        chunk_time = round(

            time.perf_counter() - start,

            3,

        )

        self.trace["chunking"] = {

            "chunks": len(self.chunks),

            "time": chunk_time,

        }

        # ----------------------------------------------------
        # Final Statistics
        # ----------------------------------------------------

        total_time = round(

            time.perf_counter() - total_start,

            3,

        )

        total_words = sum(

            chunk.word_count

            for chunk in self.chunks

        )

        average_words = round(

            total_words / len(self.chunks),

            1,

        )

        self.statistics = {

            "document": self.document_name,

            "pages": len(self.document.pages),

            "chunks": len(self.chunks),

            "words": total_words,

            "average_chunk_words": average_words,

            "index_time": total_time,

        }

        self.trace["indexing"] = {

            "total_time": total_time,

        }

        self.index_ready = True

        print("\n" + "=" * 70)
        print(" Document Indexed Successfully")
        print("=" * 70)

        print(f"Document        : {self.document_name}")
        print(f"Pages           : {self.statistics['pages']}")
        print(f"Chunks          : {self.statistics['chunks']}")
        print(f"Words           : {self.statistics['words']}")
        print(f"Average Chunk   : {average_words} words")
        print(f"Index Time      : {total_time} sec")

        print("=" * 70)

        return self.statistics
    
        # --------------------------------------------------------
    # Ask Question
    # --------------------------------------------------------

    def ask(
        self,
        question: str,
        top_k: int = 3,
    ) -> PipelineResult:
        """
        Answer a question using the indexed document.

        Pipeline:

            User Question
                    ↓
              Retrieve Chunks
                    ↓
              Build Prompt
                    ↓
                 Gemini
                    ↓
             Structured Answer

        Parameters
        ----------
        question : str

            User's question.

        top_k : int

            Number of retrieved chunks.

        Returns
        -------
        PipelineResult
        """

        if not self.index_ready:
            raise RuntimeError(
                "No indexed document found.\n"
                "Call index_document() first."
            )

        if not question.strip():
            raise ValueError(
                "Question cannot be empty."
            )

        print("\n" + "=" * 70)
        print(" SmartDocAI — Question Answering")
        print("=" * 70)

        total_start = time.perf_counter()

        # ----------------------------------------------------
        # Retrieval
        # ----------------------------------------------------

        print("\nRetrieving relevant chunks...")

        retrieval_start = time.perf_counter()

        retrieval_result = retrieve_all(

            query=question,

            chunks=self.chunks,

            top_k=top_k,

        )

        retrieval_time = round(

            time.perf_counter() - retrieval_start,

            3,

        )

        self.trace["retrieval"] = {

            "time": retrieval_time,

            "top_k": top_k,

        }

        # ----------------------------------------------------
        # Prompt Building
        # ----------------------------------------------------

        print("Building prompt...")

        prompt_start = time.perf_counter()

        prompt_package = build_prompt_from_comparison(

            query=question,

            comparison=retrieval_result,

        )

        prompt_time = round(

            time.perf_counter() - prompt_start,

            3,

        )

        self.trace["prompt_builder"] = {

            "time": prompt_time,

            "prompt_length": len(prompt_package.prompt),

        }

        # ----------------------------------------------------
        # LLM
        # ----------------------------------------------------

        print("Sending to Gemini...")

        llm_result = generate_answer(

            prompt_package.prompt

        )

        llm_time = llm_result.get(

            "latency",

            0.0,

        )

        self.trace["llm"] = {

            "time": llm_time,

            "model": llm_result.get(

                "model",

                "Unknown",

            ),

        }

        total_time = round(

            time.perf_counter() - total_start,

            3,

        )

        self.trace["pipeline"] = {

            "total_time": total_time,

        }

        # ----------------------------------------------------
        # Build PipelineResult
        # ----------------------------------------------------

        result = PipelineResult(

            question=question,

            answer=llm_result["answer"],

            confidence=llm_result["confidence"],

            source_chunk=llm_result["source_chunk"],

            reasoning=llm_result["reasoning"],

            retrieved_chunks=retrieval_result,

            prompt_package=prompt_package,

            trace=self.trace.copy(),

            total_time=total_time,

        )

        self.last_result = result

        print("\nPipeline completed successfully.")
        print(f"Total time : {total_time} sec")

        return result