from utils.PedNamesAndHashes import *
import pytest

def test_convertHashToModelName():
    assert convertHashToModelName("0xfab48bcb") == "a_f_m_fatbla_01"
    assert convertHashToModelName(0xfab48bcb) == "a_f_m_fatbla_01"
    assert convertHashToModelName("fab48bcb") == "a_f_m_fatbla_01"
    assert convertHashToModelName("fabc8245") == "UNKNOWN"
    assert convertHashToModelName("0xFAB48BCB") == "a_f_m_fatbla_01"


def test_convertModelNameToHash():
    assert convertModelNameToHash("a_f_m_fatwhite_01") == "38bad33b"
    with pytest.raises(KeyError):
        convertModelNameToHash("a_f_m_someUnknown_01")
