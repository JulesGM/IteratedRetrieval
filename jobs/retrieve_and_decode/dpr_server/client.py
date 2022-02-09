import abc
import dataclasses
from pathlib import Path
from typing import *
import sys

import fire
import grpc
from matplotlib.pyplot import title
import more_itertools
import rich

SCRIPT_DIR: Final = Path(__file__).absolute().parent
sys.path.append(str(SCRIPT_DIR / "unary"))
sys.path.append(str(SCRIPT_DIR))

import unary_pb2_grpc as pb2_grpc
import unary_pb2 as pb2
import dpr_server_utils


import server


class DPRClient(abc.ABC):
    def __init__(self, n_docs, retrieval_max_size):
        self.n_docs = n_docs
        self.retrieval_max_size = retrieval_max_size
        self.passages = self._PassageGetter(self)

    @abc.abstractmethod
    def retrieve(
        self, questions: List[str],
    ):
        pass

    @abc.abstractmethod
    def get_passages(self, ids):
        pass

    @abc.abstractmethod
    def get_len_passages(self):
        pass

    @final
    class _PassageGetter:
        """The goal of this is to make a fake `passages` 
        dict that queries the server.
        """
        def __init__(self, parent) -> None:
            self.parent = parent

        @dataclasses.dataclass
        class _FakeEntry:
            title: str
            text: str

        def __getitem__(self, id_):
            titles, texts = self.parent.get_passages([id_])
            assert len(titles) == 1, f"{len(titles) = }"
            assert len(texts) == 1, f"{len(texts) = }"
            return self._FakeEntry(title=titles[0], text=texts[0])

        def __len__(self):
            return self.parent.get_len_passages()


class SameProcClient(DPRClient):
    def __init__(self, n_docs, retrieval_max_size) -> None:
        super().__init__(n_docs, retrieval_max_size)   
        self.server = server.InMemoryServer()
    
    def retrieve(self, questions):
        return self.server.retrieve(
            questions, 
            self.n_docs,
            self.retrieval_max_size,
        )
    def get_passages(self, ids):
        titles = []
        texts = []
        for passage_id in ids:
            passage = self.server.all_passages[passage_id]
            titles.append(passage.title)
            texts.append(passage.text)
        return titles, texts

    def get_len_passages(self):
        return len(self.server.all_passages)


class UnaryClient(DPRClient):
    """
    Client for gRPC functionality
    """

    def __init__(self, n_docs, retrieval_max_size):
        super().__init__(n_docs, retrieval_max_size)

        self.host = 'localhost'
        self.server_port = 50051
        # instantiate a channel
        self.channel = grpc.insecure_channel(
            '{}:{}'.format(self.host, self.server_port)
        )
        self.stub = pb2_grpc.UnaryStub(self.channel)

    def retrieve(self, questions):
        query = pb2.RetrieveQuery()
        query.n_docs = self.n_docs

        query.questions[:] = questions
        query.retrieval_max_size = self.retrieval_max_size
        response = self.stub.Retrieve(query)
        
        scores = dpr_server_utils.decode_numpy_tensor(
            response.scores
        )

        ids = [x.strs for x in response.ids]
        titles = [x.strs for x in response.titles]
        texts = [x.strs for x in response.texts]

        return ids, scores, titles, texts

    def get_passages(self, ids):
        query = pb2.GetPassagesQuery()
        query.ids[:] = ids
        response = self.stub.GetPassages(query)
        titles = response.titles
        texts = response.texts
        return titles, texts

    def get_len_passages(self):
        response = self.stub.GetLenPassages(pb2.Empty())
        return response.len_passages


def test(remote=True, n_docs=10, retrieval_max_size=15):
    if remote:
        print("Unary Client")
        client = UnaryClient(
            n_docs=n_docs, 
            retrieval_max_size=retrieval_max_size,
        )
    else:
        print("Local Client")
        client = SameProcClient(
            n_docs=n_docs, 
            retrieval_max_size=retrieval_max_size,
        )

    print("Client: Doing the request")
    questions = ["what is the capital of France"]
    ids, scores, titles, texts = client.retrieve(
        questions=questions
    )
    for i, (question, titles, texts) in enumerate(
        more_itertools.zip_equal(questions, titles, texts)
    ):
        rich.print(f"{i} - [green bold]Question[/]: {question}")
        for j, (title, passage) in enumerate(zip(titles, texts)):
            rich.print(f"{i} - [green bold]Title {j}[/]:   {title}")
            rich.print(f"{i} - [green bold]Passage {j}[/]: {passage}")
            print("")
        print("#" * 80 + "\n")

    print("\n" + "#" * 80 + "\n")
    rich.print("[red bold]Testing GetPassages[/]")
    ids = ["321", "123"]
    titles, text = client.get_passages(ids)
    for i, (title, text) in enumerate(
        more_itertools.zip_equal(titles, text)
    ):
        rich.print(f"[green bold]Title[/]: {title}")
        rich.print(f"[green bold]Text[/]:")
        rich.print(text)
        print("\n")
    print("\n" + "#" * 80 + "\n")



if __name__ == '__main__':
    fire.Fire(test)