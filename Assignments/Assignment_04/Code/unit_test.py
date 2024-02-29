import numpy as np
from main import PoissonBlend


def test_PoissonBlend():
    source = np.array([[.8,.6,.6,.6],
                       [.6,.6,.2,.6],
                       [.6,.8,.6,.6]])
    mask = np.array([[0,0,0,0],
                        [0,1,1,0],
                        [0,0,0,0]])
    target = np.array([[.2,.5,.2,.2],
                        [.7,.7,.7,.7],
                        [.9,.9,.8,.9]])
    
    source = np.stack([source, source, source], axis=2)
    mask = np.stack([mask, mask, mask], axis=2)
    target = np.stack([target, target, target], axis=2)
    
    result = PoissonBlend(source, mask, target, isMix=False)[:, :, 0]

    expected = np.array([[0.2, 0.5, 0.2, 0.2],
                            [0.7, 0.62, 0.18, 0.7],
                            [0.9, 0.9, 0.8, 0.9]])
    
    assert np.allclose(result, expected, atol=1e-2), f"Expected {expected}, but got {result}"
    

