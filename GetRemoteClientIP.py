from streamlit import runtime
from streamlit.runtime.scriptrunner import get_script_run_ctx
import streamlit as st
import subprocess

def get_remote_ip() -> str:
    """Get remote ip."""
    try:
        ctx = get_script_run_ctx()
        if ctx is None:
            return None

        session_info = runtime.get_instance().get_client(ctx.session_id)
        if session_info is None:
            return None
    except Exception as e:
        return None

    return session_info.request.remote_ip

def GetUserName(ip) -> str:
    if len(ip) >= 8:
        try:
            username = subprocess.check_output('wmic.exe /node:"' + ip + '" ComputerSystem Get UserName')
            username = username.split('UserName')
        except Exception as ex:
            print(ex)
            username = ip
    else:
        username = 'admin'

    return username
