from concurrent import futures
from pathlib import Path
import sys
import threading
from typing import *
import time
import contextlib
import more_itertools
import rich

SCRIPT_DIR: Final = Path(__file__).absolute().parent
sys.path.append(str(SCRIPT_DIR / "unary"))
sys.path.append(str(SCRIPT_DIR))

import fire
import grpc
import unary.unary_pb2_grpc as pb2_grpc
import unary.unary_pb2 as pb2
import dpr_server_utils

ROOT_PATH: Final = SCRIPT_DIR.parent.parent.parent
GAR_PATH: Final = ROOT_PATH / "GAR" / "gar"
DPR_PATH: Final = ROOT_PATH / "DPR"
RETRIEVE_AND_DECODE_PATH: Final = ROOT_PATH / "jobs" / "retrieve_and_decode"
CONF_PATH: Final = DPR_PATH / "conf"
sys.path.insert(0, str(GAR_PATH))
sys.path.insert(0, str(DPR_PATH))
sys.path.insert(0, str(RETRIEVE_AND_DECODE_PATH))

import common_retriever  # type: ignore
import iterated_utils as utils  # type: ignore


with dpr_server_utils.timeit("Loading pytorch"):
    import torch


class InMemoryServer:
    def __init__(self):
        self.dpr_cfg: Final = common_retriever.build_cfg(str(DPR_PATH / "conf"))
        with dpr_server_utils.timeit("Loading retriever."):
            with dpr_server_utils.timeit("-> Building it."):
                (
                    self.retriever, 
                    self.all_passages, 
                    self.special_query_token
                ) = common_retriever.build_retriever(
                    self.dpr_cfg,
                    ROOT_PATH / "jobs" / "retrieve_and_decode" / "cache" 
                )

            with dpr_server_utils.timeit("-> Sending the index to the GPU."):
                self.retriever.index.index = common_retriever.faiss_to_gpu(
                    self.retriever.index.index,
                )

            self.retriever_lock: Final = threading.Lock()

    def retrieve(self, questions, n_docs, retrieval_max_size):
        with self.retriever_lock:
            top_ids, top_scores = common_retriever.retrieve(
                retriever=self.retriever,
                all_passages=self.all_passages,
                questions=questions,
                special_query_token=self.special_query_token,
                n_docs=n_docs,
                retrieval_max_size=retrieval_max_size,
            )
        texts: Final = []
        titles: Final = []
        for top_ids_batch in top_ids:
            texts.append(
                [self.all_passages[idx].text for idx in top_ids_batch]
            )
            titles.append(
                [self.all_passages[idx].title for idx in top_ids_batch]
            )

        return top_ids, top_scores, titles, texts


class UnaryService(pb2_grpc.UnaryServicer):
    def __init__(self, *args, **kwargs):
        self.retriever = InMemoryServer()
        print("Done building the index.")

    def Retrieve(self, request, context):
        top_ids, top_scores, titles, texts = self.retriever.retrieve(
            request.questions, 
            request.n_docs, 
            request.retrieval_max_size,
        )
        
        score_bytes = dpr_server_utils.encode_numpy_tensor(top_scores)
        response = pb2.RetrieveResponse()
        response.scores = score_bytes
        for batch in top_ids:
            los = pb2.ListOfStrings()
            los.strs[:] = batch
            response.ids.append(los)
        for batch in titles:
            los = pb2.ListOfStrings()
            los.strs[:] = batch
            response.titles.append(los)
        for batch in texts:
            los = pb2.ListOfStrings()
            los.strs[:] = batch
            response.texts.append(los)
        return response

    def GetPassages(self, request, context):
        response = pb2.GetPassagesResponse()
        for passage_id in request.ids:
            passage = self.retriever.all_passages[passage_id]
            response.titles.append(passage.title)
            response.texts.append(passage.text)
        return response

    def GetLenPassages(self, request, context):
        return pb2.GetLenPassagesResponse(
            len=len(self.retriever.all_passages)
        )


def serve(port=50051, num_workers=5):
    print("Server: grpc.server(...)")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=num_workers))
    print("Server: pb2_grpc.add_UnaryServicer_to_server")
    pb2_grpc.add_UnaryServicer_to_server(UnaryService(), server)
    print("Server: server.add_insecure_port(...)")
    server.add_insecure_port(f'[::]:{port}')
    print("server.start()")
    server.start()
    print("server.wait_for_termination")
    server.wait_for_termination()


if __name__ == '__main__':
    fire.Fire(serve)