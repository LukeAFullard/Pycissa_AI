import numpy as np
from pycissa.postprocessing.statistics.average_mutual_information import (
    calculate_average_mutual_information_Stergiou,
    AMI_Thomas
)

def test_stergiou_int_L():
    data = np.sin(np.linspace(0, 10, 100))
    tau, v_ami = calculate_average_mutual_information_Stergiou(data, 10)
    assert tau is not None
    assert v_ami is not None

def test_stergiou_array_L():
    data = np.sin(np.linspace(0, 10, 100))
    L_array = np.cos(np.linspace(0, 10, 100))
    ami = calculate_average_mutual_information_Stergiou(data, L_array)
    assert isinstance(ami, (float, np.floating))
    assert ami >= 0

def test_thomas_int_L():
    data = np.sin(np.linspace(0, 10, 100))
    tau, v_ami = AMI_Thomas(data, 10)
    assert tau is not None
    assert v_ami is not None

def test_thomas_array_L():
    data = np.sin(np.linspace(0, 10, 100))
    L_array = np.cos(np.linspace(0, 10, 100))
    ami = AMI_Thomas(data, L_array)
    assert isinstance(ami, (float, np.floating))
    assert ami >= 0
