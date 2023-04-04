import main
import os
import typing
from flytekit import workflow
from main import Hyperparameters
from main import run_workflow

_wf_outputs=typing.NamedTuple("WfOutputs",run_workflow_0=main.PneumoniaTrainer)
@workflow
def wf_26(_wf_args:Hyperparameters)->_wf_outputs:
	run_workflow_o0_=run_workflow(hp=_wf_args)
	return _wf_outputs(run_workflow_o0_)