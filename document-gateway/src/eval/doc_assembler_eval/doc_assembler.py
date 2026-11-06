import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

import requests
from document_common.utils.logger import Logger

from api_gateway.models.document import SectionResponse
from src.eval.doc_assembler_eval.models import DocRef, GroundTruth
from src.eval.utils.utils import load_documents


class DocAssembler:
    """Class to generate documents using our tool"""
    def __init__(
            self,
            api_base_url: str, # This is the api of the running document app
            documents_dir: str, # This is where the generated documents will be stored
            ground_truth_path: str,
            document_dir: str | List[str],
            n_document_per_gt: int = 2,
            oracle: bool = False,
            max_assembly_retries: int = 3
        ):
        self.logger = Logger().get_logger("eval.main")

        self.api_base_url = api_base_url.rstrip("/")

        self.documents_dir = Path(documents_dir)
        self.documents_dir.mkdir(exist_ok=True)
        self.ground_truth_path = Path(ground_truth_path)

        self.n_document_per_gt = n_document_per_gt
        self.oracle = oracle
        self.max_assembly_retries = max_assembly_retries

        self.doc_refs = load_documents(document_dir)

    def generate_documents(self, documents: List[DocRef], template_id: str, subject: str | None, context: Dict[str, Any], num_documents: int) -> List[str]:
        """Generate multiple documents using the imported documents."""
        job_ids: List[str] = []

        # Convert documents to the format expected by generate-document
        doc_refs = [
            {
                "coreDataId": doc["coreDataId"],
                "coreType": doc["coreType"],
                "coreDataSource": doc["coreDataSource"]
            }
            for doc in documents
        ]

        self.logger.debug(f"Generating {num_documents} documents with {len(doc_refs)} documents")

        if subject:
            context["subject"] = subject

        for i in range(num_documents):
            url = f"{self.api_base_url}/generate-document"
            payload = {
                "templateId": template_id,
                "docs": doc_refs,
                "context": context,
                "flush": False
            }

            self.logger.info(f"Generating document {i+1}/{num_documents}...")
            try:
                response = requests.post(url, json=payload)

                if response.status_code == 202:
                    job_id = response.json().get("jobId")
                    job_ids.append(job_id)
                    self.logger.info(f"document {i+1} assembly started (Job ID: {job_id})")
                else:
                    self.logger.error(f"Failed to start document {i+1}: {response.status_code} - {response.text}")
                    continue
            except Exception as e:
                self.logger.error(f"Error generating document {i+1}: {e}", exc_info=True)
                continue

            # Small delay to avoid overwhelming the API
            time.sleep(1)

        self.logger.info(f"Started assembly of {len(job_ids)} documents")
        return job_ids

    def wait_for_documents(self, job_ids: List[str], timeout: int = 18000) -> Dict[str, Any]:
        """Wait for all documents to complete and return their data."""
        completed_documents: Dict[str, Any] = {}
        start_time = time.time()

        while time.time() - start_time < timeout:
            all_finalized = True

            for job_id in job_ids:
                if job_id in completed_documents:
                    continue

                url = f"{self.api_base_url}/documents/{job_id}"
                try:
                    response = requests.get(url)

                    if response.status_code == 200:
                        document_data = response.json()
                        status = document_data.get("status")

                        if status == "completed":
                            completed_documents[job_id] = document_data
                            self.logger.info(f"document {job_id} completed")
                        elif status == "failed":
                            self.logger.error(f"document {job_id} failed")
                        else:
                            all_finalized = False
                            self.logger.debug(f"document {job_id} status: {status}")
                    else:
                        self.logger.error(f"Failed to check document {job_id}: {response.status_code}")
                        all_finalized = False
                except Exception as e:
                    self.logger.error(f"Error checking document {job_id} status: {e}", exc_info=True)
                    all_finalized = False

            if all_finalized:
                self.logger.info("All documents finalized")
                return completed_documents

            time.sleep(10)  # Wait 10 seconds before checking again

        self.logger.info("Timeout waiting for documents")
        return completed_documents

    def section_to_text(self, section: SectionResponse, level:int=1) -> str:
        document_text = ""
        if section.generated_content:
            document_text += f"{'#'*level} {section.title}\n"
            document_text += f"{section.generated_content}\n"
        if section.sections:
            sub_sections = sorted(section.sections, key=lambda x: x.order)
            for sub_s in sub_sections:
                document_text += self.section_to_text(section=sub_s, level=level+1)
        return document_text

    def document_to_text(self, document_data: dict[str, Any]):
        sections = [SectionResponse.model_validate(s) for s in document_data['sections']]
        sections = sorted(sections, key=lambda x: x.order)
        document_text = ""
        for section in sections:
            document_text += self.section_to_text(section)
        
        return document_text

    def save_documents(self, documents: Dict[str, Any]) -> None:
        """Save completed documents to files."""
        for job_id, document_data in documents.items():
            filename = f"{job_id}.md"
            filepath = self.documents_dir / filename

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(self.document_to_text(document_data))

            self.logger.info(f"Saved document {job_id} to {filepath}")

    def save_gt_document_mapping(self, gt_document_mapping: Dict[str, List[str]]) -> None:
        """Save the ground truth to document mapping to a JSON file with a UUID filename."""
        filename = f"{uuid.uuid4()}.json"
        filepath = self.documents_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(gt_document_mapping, f, indent=4)

        self.logger.info(f"Saved gt_document_mapping to {filepath}")
    
    def run(self) -> dict[str, List[str]]:
        """Run the complete dataset evaluation pipeline."""
        self.logger.info("Starting dataset evaluation...")

        try:
            # Step 1: Load ground truth
            self.logger.info("Loading ground truth...")
            ground_truth = GroundTruth.load_ground_truth(ground_truth_path=self.ground_truth_path)
            self.logger.info(f"Loaded {len(ground_truth)} ground truth entries")
            
            # Step 2: Generate documents for each ground truth entry
            self.logger.info("Generating documents...")
            gt_document_mapping: Dict[str, list[str]] = {}
            completed_documents: Dict[str, Any] = {}
            for gt in ground_truth:
                if self.oracle: # To be implemented later for now it is not used.
                    gold_docs = gt.gold_documents
                    docs_to_use = [d for d in self.doc_refs if d['filename'] in gold_docs]
                else:
                    docs_to_use = self.doc_refs

                # Generate documents with retry logic
                completed_job_ids: List[str] = []
                for attempt in range(self.max_assembly_retries):
                    new_ids = self.generate_documents(
                        docs_to_use, 
                        gt.template, 
                        gt.subject, 
                        gt.context or {}, 
                        self.n_document_per_gt - len(completed_job_ids)
                    )
                    self.logger.info(f"Started assembly of {len(new_ids)} documents for ground truth {gt.id}")

                    self.logger.info(f"Waiting for documents (attempt {attempt + 1}/{self.max_assembly_retries})...")
                    new_documents = self.wait_for_documents(new_ids)

                    completed_job_ids.extend(new_documents.keys())
                    completed_documents = completed_documents | new_documents

                    if len(completed_job_ids) >= self.n_document_per_gt:
                        break
                if len(completed_job_ids) < self.n_document_per_gt:
                    raise Exception(f"assembly failed after {self.max_assembly_retries} tries")
                
                gt_document_mapping[gt.id] = [id for id in completed_job_ids]

            self.logger.info(f"Completed {len(completed_documents.keys())} documents")

            # Step 5: Save documents
            self.logger.info("Saving documents...")
            self.save_documents(completed_documents)

            # Save the ground truth to document mapping
            self.logger.info("And the mapping to the matching gt")
            self.save_gt_document_mapping(gt_document_mapping)

            return gt_document_mapping
        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}", exc_info=True)
            raise
