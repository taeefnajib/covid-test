import project
import os
import typing
from flytekit import workflow
from project.wf_48_318.main import Hyperparameters
from project.wf_48_318.main import run_workflow

_wf_outputs=typing.NamedTuple("WfOutputs",run_workflow_0=project.wf_48_318.main.PneumoniaTrainer)
@workflow
def wf_48(_wf_args:Hyperparameters)->_wf_outputs:
	run_workflow_o0_=run_workflow(hp=_wf_args)
	return _wf_outputs(run_workflow_o0_)