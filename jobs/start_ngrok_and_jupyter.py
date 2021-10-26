#! /usr/bin/env python3
import abc
import pathlib
import os
import re
import subprocess
import time
import urllib3
import yaml

import fire
import requests

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent

NGROK_PORT_DEFAULT = 8889
NGROK_START_MAX_RETRIES = 15
NGROK_WAIT_TIME = 1
JUPYTER_MAX_RETRIES = 5
JUPYTER_WAIT_TIME = 1
DEFAULT_NOTEBOOK_DIR = PROJECT_ROOT  # Assumes the script is at PROJECT_ROOT/jobs/here.py
DEFAULT_GUI = "lab"

def safe_load_yaml(path):
    with open(path) as fin:
        return yaml.safe_load(fin)


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

    @property
    @abc.abstractmethod
    def max_retries(self):
        pass

    @property
    @abc.abstractmethod
    def wait_time(self):
        pass

    class _FailedGettingPayload:
        pass
    failed_getting_payload = _FailedGettingPayload()

    def payload_is_ok(self, payload):
        return payload is not self.failed_getting_payload

    def maybe_start(self, *start_process_args, **start_process_kwargs):
        payload = self.is_online()
        if not self.payload_is_ok(payload):
            self.start_process(*start_process_args, **start_process_kwargs)
        else:
            print(f"There is already a {self.name} process.")
        return payload

    def wait_until_payload(self, payload):
        if not self.payload_is_ok(payload):
            payload = self._get_payload()
            if not self.payload_is_ok(payload):
                print(f"{self.name}.wait_until_payload: Failed to connect to {self.name.capitalize()} server.")
                return self.failed_getting_payload
        return payload

    def start_process(self, *args, **kwargs):
        print(f"Starting a new {self.name} process.")
        subprocess.Popen(
                self.create_command(*args, **kwargs),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

    def _get_payload(self):
        payload = self.failed_getting_payload
        for _ in range(self.max_retries):
            payload = self.is_online()
            if self.payload_is_ok(payload):
                break
            print(f"{self.name}.get_payload: Waiting process to start.")
            print(f"{self.name}.get_payload: sleeping for {self.wait_time}s")
            time.sleep(self.wait_time)
        
        return payload

    @abc.abstractmethod
    def is_online(self):
        pass

    @abc.abstractmethod
    def create_command(self):
        pass


class NgrokContext(ContextBase):
    name = "ngrok"
    max_retries = NGROK_START_MAX_RETRIES
    wait_time = NGROK_WAIT_TIME

    def is_online(self):
        try: 
            r = requests.get("http://127.0.0.1:4040/api/tunnels")
        except (
            requests.exceptions.ConnectionError, 
            urllib3.exceptions.NewConnectionError,
            ):
            return self.failed_getting_payload
        ngrok_info = r.json()
        ngrok_tunnels = ngrok_info["tunnels"]
        if len(ngrok_tunnels) < 1:
            return self.failed_getting_payload
        return ngrok_tunnels[0]["public_url"]

    def create_command(self, port):
        return ["ngrok", "http", str(port)]


class GUIContext(ContextBase, abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def format():
        pass


class JupyterMixin(GUIContext):
    def _is_online(self, util_name):
        jupyter_output = subprocess.check_output(
            ["jupyter", util_name, "list"]
        )
        jupyter_lines = jupyter_output.decode().strip().split("\n")
        jupyter_token = self.failed_getting_payload
        if len(jupyter_lines) > 1:
            assert len(jupyter_lines) == 2, f"\"{jupyter_lines}\""
            jupyter_token = jupyter_lines[1].split()[0].split("=", 1)[1]

        return jupyter_token

    def _create_command(self, util_name, ngrok_port, dir):
        return [
            "jupyter", 
            util_name, 
            "--no-browser", 
            f"--port={ngrok_port}",
            f"--notebook-dir={dir}",
        ]

    @staticmethod
    def format(address, gui_payload):
        return f"{address}/?token={gui_payload}"


class NotebookContext(JupyterMixin):
    name = "jupyter-notebook"
    max_retries = JUPYTER_MAX_RETRIES
    wait_time = JUPYTER_WAIT_TIME

    def is_online(self):
        return self._is_online("notebook")

    def create_command(self, ngrok_port, notebook_dir):
        return self._create_command(
            "notebook", ngrok_port, notebook_dir
        )


class LabContext(JupyterMixin):
    name = "jupyter-lab"
    max_retries = JUPYTER_MAX_RETRIES
    wait_time = JUPYTER_WAIT_TIME

    def is_online(self):
        return self._is_online("lab")

    def create_command(self, ngrok_port, notebook_dir):
        return self._create_command("lab", ngrok_port, notebook_dir)


class CodeServerContext(GUIContext):
    name = "code-server"
    max_retries = JUPYTER_MAX_RETRIES
    wait_time = JUPYTER_WAIT_TIME
    config_path = pathlib.Path(
        os.environ["HOME"]
    ) / ".config/code-server/config.yaml"

    def is_online(self):
        output = subprocess.check_output(
            ["ps", "-o", "cmd="]
        )

        lines = output.decode().strip().split("\n")
        print(lines)
        
        token = None
        matches = [re.match(r".*node.*code\-server.*", line) for line in lines]
        print(matches)
        print()
        if any(matches):
            token = safe_load_yaml(self.config_path)["password"]
        return token


    def create_command(self, notebook_dir):
        return ["code-server", notebook_dir]

    @staticmethod
    def format(address, gui_payload):
        return f"{address} \"{gui_payload}\""



def main(gui=DEFAULT_GUI, ngrok_port=NGROK_PORT_DEFAULT, notebook_dir=DEFAULT_NOTEBOOK_DIR):
    print(f"{gui = }")
    print(f"{ngrok_port = }")
    print(f"{notebook_dir = }")
    print(f"")

    guis = {
        "notebook": dict(
            init=NotebookContext, 
            init_args=(), 
            init_kwargs=dict(),
            start_process_args=(ngrok_port, notebook_dir),
            start_process_kwargs=dict(),
        ),
        "lab": dict(
            init=LabContext, 
            init_args=(), 
            init_kwargs=dict(),
            start_process_args=(ngrok_port, notebook_dir),
            start_process_kwargs=dict(),
        ),
        "code-server": dict(
            init=CodeServerContext, 
            init_args=(), 
            init_kwargs=dict(),
            start_process_args=(notebook_dir,),
            start_process_kwargs=dict(),
        )
    }

    configs = dict(
        ngrok=dict(
            init=NgrokContext, 
            init_args=(),
            init_kwargs=dict(),
            start_process_args=(ngrok_port,),
            start_process_kwargs=dict(),
        ),
        gui=guis[gui],
    )
    
    payloads = dict()
    contexts = dict()

    # Init and maybe start
    for name, config in configs.items():
        # Init
        contexts[name] = config["init"](
            *config["init_args"], 
            **config["init_kwargs"]
        )
        payloads[name] = contexts[name].maybe_start(
            *config["start_process_args"], 
            **config["start_process_kwargs"],
        )

    # Wait until payload
    for name, payload in payloads.items():
        payloads[name] = contexts[name].wait_until_payload(payload)
        if not contexts[name].payload_is_ok(payloads[name]):
            raise RuntimeError("Payload failed: {name}")

    # Put things together
    output = contexts["gui"].format(payloads["ngrok"], payloads["gui"])
    print(f"\n{output}")
    

if __name__ == "__main__":
    fire.Fire(main)
