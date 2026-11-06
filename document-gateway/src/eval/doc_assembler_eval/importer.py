import time
from typing import List

import requests
from document_common.utils.logger import Logger

from src.eval.doc_assembler_eval.models import DocRef
from src.eval.utils.utils import load_documents


class Importer:
    def __init__(
            self,
            api_base_url: str, # This is the api of the running document app
            documents_dir: List[str] | str, # These will be check for files that are in .txt, the name will be the id
            max_importation_retries: int = 3
        ):
        self.logger = Logger().get_logger("eval.main")

        self.api_base_url = api_base_url.rstrip("/")
        self.documents_dir = documents_dir
        self.max_importation_retries = max_importation_retries

    def import_documents(self, documents: List[DocRef]) -> bool:
        """Send documents to the import endpoint."""
        url = f"{self.api_base_url}/import"
        payload = {"docs": documents}

        self.logger.info(f"importing {len(documents)} documents...")
        try:
            response = requests.post(url, json=payload)

            if response.status_code == 202:
                self.logger.info("Documents importation started successfully")
                return True
            else:
                self.logger.error(f"Failed to start importation: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            self.logger.error(f"Error during document importation request: {e}", exc_info=True)
            return False

    def wait_for_importation(self, documents: List[DocRef], timeout: int = 1200) -> List[DocRef]:
        """Wait for all documents to be imported."""
        start_time = time.time()
        
        documents_imported: List[DocRef] = []
        while time.time() - start_time < timeout:
            all_finalized = True

            for doc in [d for d in documents if d not in documents_imported]: # We only check non imported files.
                doc_id = doc["coreDataId"]
                url = f"{self.api_base_url}/import/{doc_id}/status"
                try:
                    response = requests.get(url)

                    if response.status_code == 200:
                        status_data = response.json()
                        status = status_data.get("status")
                        self.logger.info(f"Document {doc_id}: {status}")

                        if status == "imported":
                            documents_imported.append(doc)
                            self.logger.info(f"imported Document {doc_id}")
                            continue
                        elif status in ["pending", "processing"]:
                            all_finalized = False
                        elif status == "failed":
                            self.logger.warning(f"Document {doc_id} failed importation")
                            continue
                        else:  # unknown status
                            self.logger.warning(f"Document {doc_id} has unknown status: {status}")
                            all_finalized = False
                    else:
                        self.logger.error(f"Failed to check status for {doc_id}: {response.status_code}")
                        all_finalized = False
                        continue
                except Exception as e:
                    self.logger.error(f"Error checking status for document {doc_id}: {e}", exc_info=True)
                    all_finalized = False
                    continue

            if all_finalized:
                self.logger.info("All documents finalized")
                return documents_imported

            time.sleep(5)  # Wait 5 seconds before checking again

        self.logger.info("Timeout waiting for document importation")
        raise TimeoutError(f"Waiting for importation timed out after {timeout} seconds. {len(documents_imported)}/{len(documents)} Files imported.")
    
    def run(self) -> None:
        """Run the complete dataset evaluation pipeline."""
        self.logger.info("Starting dataset evaluation...")

        try:
            # Step 1: Load documents
            self.logger.info("Loading documents...")
            documents = load_documents(self.documents_dir)
            self.logger.info(f"Loaded {len(documents)} documents")

            # Step 2: import documents with retries
            self.logger.info("importing documents...")

            remaining_docs = documents
            for attempt in range(self.max_importation_retries):
                self.logger.info(f"importing documents (attempt {attempt + 1}/{self.max_importation_retries})...")
                # TODO : faut maj les uuids
                if not self.import_documents(remaining_docs):
                    continue
                doc_imported = self.wait_for_importation(remaining_docs)

                self.logger.info(f"{len(doc_imported)} Documents imported")
                
                remaining_docs = [doc for doc in remaining_docs if doc not in doc_imported]
                if not remaining_docs:
                    continue
            if remaining_docs:
                raise Exception(f"importation failed after {self.max_importation_retries} tries...")
            

        except Exception as e:
            self.logger.error(f"importation failed: {e}", exc_info=True)
            raise
