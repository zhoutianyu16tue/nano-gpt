# Nano-gpt
This repository tried to reproduce the nanoGPT implementation by Andrej Karpathy.  
Definitely check out his content on [YouTube](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=6039s) and [GitHub](https://github.com/karpathy/nanoGPT)  

Instead of Nvidia GPU, I managed to port Andrej's code to Huawei Ascend AI accelerator.
I made the following changes,
* I added mixed precision to the pytorch code so the training can finish reasonably fast;
* I used [Ascend graph compiler](https://www.hiascend.com/document/detail/en/canncommercial/600/inferapplicationdev/atctool/atlasatc_16_0007.html) to generate a more efficient graph representation; so the inference can work.

## Training Result
It took less than 6 minutes for the 10M-parameter model to converge, after 5000 iterations.
Andrej said in his [nanoGPT video](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=6039s)
that the same 10M-parameter model converged in his A100 GPU after around 15 minutes.
```
step 500: train loss 2.0106, val loss 2.0910
step 1000: train loss 1.6114, val loss 1.7706
step 1500: train loss 1.4453, val loss 1.6341
step 2000: train loss 1.3443, val loss 1.5676
step 2500: train loss 1.2806, val loss 1.5285
step 3000: train loss 1.2340, val loss 1.4987
step 3500: train loss 1.1867, val loss 1.4860
step 4000: train loss 1.1474, val loss 1.4831
step 4500: train loss 1.1080, val loss 1.4872
step 4999: train loss 1.0767, val loss 1.4866
```

## Inference Result
### Graph Compilation
```
atc --model nanoGPT.onnx --framework 5 --soc_version Ascend310 --output nanoGPT
```
### Text Generation
After 5000 iterations, the generated text starts to make some sense and look like English text with some Shakespeare vibe.
```
You have, the rob a breather Hastings,
That your tongue vows betow him friends:
I think, good Montague,
Master have disposed fair made and half
To saved one bloody than my life of the fiery
To unless the rile whom could be so hurs,
Say and yet, Fitzwards, siring of his virtuous;
Helm him was perform to the deople much growth
Of his hands: set the jewel, that the orders.

BISHOP OF ELY:
Twelcome is the time
From our shepherds hideed in the bride.

QUEEN MARGARET:
Tlet he scave me this sovereignty flatterers;
I think it with my end to I thee and wars,
In poor God and yet as to see thee;
Of all the way that I have mudded mine sun
And speak crown'd in jades by the nrice of tears.
Where, by my prisonous pawer that, by thou wast
Seeming their gatesy have bleeding drength.
Here lose that we indeed so brows by her own
And unatted unborneign a womb! What a man?
Methought despection and had been his life,
Proud he some severeigns of second crimm,
Open his monument blood from our honour hither.
```
## Environment
Training: PyTorch-1.8.1 from [Ascendhub](https://ascendhub.huawei.com/#/detail/pytorch-modelzoo) with tag **22.0.0-1.8.1**.  
Inference: CANN from [Ascendhub](https://ascendhub.huawei.com/#/detail/infer-modelzoo) with tag **22.0.0**
