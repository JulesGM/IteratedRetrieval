import abc
import os
import subprocess
import time
import urllib3

import fire
import requests

NGROK_PORT_DEFAULT = 8889
NGROK_START_MAX_RETRIES = 15
NGROK_WAIT_TIME = 1
JUPYTER_MAX_RETRIES = 5
JUPYTER_WAIT_TIME = 1
DEFAULT_NOTEBOOK_DIR = os.environ["HOME"]

def clear():
    subprocess.check_output(["reset"])


def poll_and_message(name, proc_obj):
    result = proc_obj.poll()
    if result is None:
        print(f"{name} is running. PID: {proc_obj.pid}")
    else:
        print(f"{name} is not running! Result: {result}, type: {type(result)}")


class ContextBase(abc.ABC):
    @property
    @abc.abstractmethod
    def name(self):
        pass

    def maybe_start(self, *start_process_args, **start_process_kwargs):
        payload = self.is_online()
        if not payload:
            self.start_process(*start_process_args, **start_process_kwargs)
        else:
            print(f"There is already a {self.name} process.")
        return payload

    def wait_until_payload(self, payload):
        if payload is None:
            payload = self.get_payload()
            if payload is None:
                print(f"Failed to connect to {self.name.capitalize()} server.")
                return 1
        return payload

    def launch_process(self, command):
        print(f"Starting a new {self.name} process.")
        subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

    @abc.abstractmethod
    def is_online(self):
        pass

    @abc.abstractmethod
    def get_payload(self):
        pass
    
    @abc.abstractmethod
    def start_process():
        pass


class NgrokContext(ContextBase):
    name = "ngrok"
    
    def is_online(self):
        try: 
            r = requests.get("http://127.0.0.1:4040/api/tunnels")
        except (
            requests.exceptions.ConnectionError, 
            urllib3.exceptions.NewConnectionError,
            ):
            return None
        ngrok_info = r.json()
        ngrok_tunnels = ngrok_info["tunnels"]
        if len(ngrok_tunnels) < 1:
            return None
        return ngrok_tunnels[0]["public_url"]

    def get_payload(self):
        ngrok_url = None
        for i in range(NGROK_START_MAX_RETRIES):
            ngrok_url = self.ngrok_is_online()
            if ngrok_url:
                break
            print("Waiting for ngrok to start.")
            print("sleeping get_ngrok_url")
            time.sleep(NGROK_WAIT_TIME)
        return ngrok_url

    def start_process(self, port):
        super.start_process(["ngrok", "http", str(port)])


class JupyterContext(ContextBase):
    name = "jupyter"

    def __init__(self, use_lab):
        super().__init__()
        self.use_lab = use_lab

    def is_online(self):
        jupyter_output = subprocess.check_output(
            ["jupyter", "lab" if self.use_lab else "notebook", "list"]
        )
        jupyter_lines = jupyter_output.decode().strip().split("\n")
        jupyter_token = None
        if len(jupyter_lines) > 1:
            assert len(jupyter_lines) == 2, f"\"{jupyter_lines}\""
            jupyter_token = jupyter_lines[1].split()[0].split("=", 1)[1]

        return jupyter_token

    def start_process(self, ngrok_port, notebook_dir):
        super.start_process([
                    "jupyter", 
                    "lab" if self.use_lab else "notebook", 
                    "--no-browser", 
                    f"--port={ngrok_port}",
                    f"--notebook-dir={notebook_dir}",
        ])

    def get_payload(self):
        jupyter_token = None
        for _ in range(JUPYTER_MAX_RETRIES):
            jupyter_token = self.is_online()
            if jupyter_token:
                break
            print("Waiting for the jupyter notebook to start.")
            print("sleeping get_jupyter_token")
            time.sleep(JUPYTER_WAIT_TIME)
        return jupyter_token


def main(ngrok_port=NGROK_PORT_DEFAULT, notebook_dir=DEFAULT_NOTEBOOK_DIR, lab=False):
    configs = dict(
        ngrok=dict(
            init=NgrokContext, 
            init_args=(),
            init_kwargs=dict(),
            start_process_args=(ngrok_port,),
            start_process_kwargs=dict(),
        ),
        jupyter=dict(
            init=JupyterContext, 
            init_args=(lab,), 
            init_kwargs=dict(),
            start_process_args=(ngrok_port, notebook_dir),
            start_process_kwargs=dict(),
        ),
    )
    payloads = dict()
    contexts = dict()

    # Init and maybe start
    for name, config in configs.items():
        # Init
        contexts[name] = config["init"](*config["init_args"])
        payloads[name] = contexts[name].maybe_start(
            *config["start_process_args"], 
            **config["start_process_kwargs"],
        )

    # Wait until payload
    for name, payload in payloads.items():
        payloads[name] = contexts[name].wait_until_payload(payload)

    # Put things together
    ngrok_url = payloads["ngrok"]
    jupyter_token = payloads["jupyter"]
    print(f"\n{ngrok_url}/?token={jupyter_token}")
    

if __name__ == "__main__":
    fire.Fire(main)