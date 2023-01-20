# Nano-gpt
This repository tried to reproduce the nanoGPT implementation by Andrej Karpathy.  
Definitely check out his content on [YouTube](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=6039s) and [Github](https://github.com/karpathy/nanoGPT)  

Instead of Nvidia GPU, I managed to port Andrej's code to Huawei Ascend AI accelerator.
I made the following changes,
* I added mixed precision to the pytorch code so the training can finish reasonably fast;
* I used [Ascend graph compiler](https://www.hiascend.com/document/detail/en/canncommercial/600/inferapplicationdev/atctool/atlasatc_16_0007.html) to generate a more efficient graph representation; so the inference can work.

## Training Result
It took around 13 minutes for the 10M-parameter model to converge, after 5000 iterations.
Andrej said in his [nanoGPT video](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=6039s)
that the same 10M-parameter model converged in his A100 GPU after around 15min.
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

## Tnference Result
### Graph Compilation
```
atc --model nanoGPT.onnx --framework 5 --soc_version Ascend310 --output nanoGPT
```
### Text Generation
After 5000 iterations, the generated text starts to make some sense and look like English text with some Shakespeare vibe.
```
LEONTES:
And so person the lubstant.

LEONTES:
Go,
Master, you had; wonded to Paris Camillo,
Beaut, that honour may leave the heart be
For will I wake for a vewy touch that he hath
He hath we might behind you are one known,
To hear my lord; for, being confinied within
Above it for my noble modesty,
And she bite us or earth to thee widow
Into you advocation
Death preserved at the house.

AUTOLYCUS:
Marry, I cannot come, you have been to age,
I can, that raised our praises what he can was,
Your doth quake itself thee in my see
You have pass'd fested mine own present mine hollow,
All makes that take you for Irish better mother,
Which now or but show'd the spirits whole God looks
Coming wait hath ruin'd vile all men more flies,
And where Lord Hastings in your spoils world,
Though shalt that fatal money, here be crown'd
The dready of his plucking Richmond:' if Henry had cold
And left thy last sons that the treacherous tirms:
When have I must I make my words I can not,
I see thee had vain. 
```
