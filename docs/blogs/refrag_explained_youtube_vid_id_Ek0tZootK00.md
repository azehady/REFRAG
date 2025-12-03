Youtube Video:
URL: https://www.youtube.com/watch?v=Ek0tZootK00
Description: REFRAG from Meta Superintelligence Labs is a SUPER exciting breakthrough that may spark the second summer of Vector Databases! REFRAG illustrates how Database Systems are becoming even more integral to LLM inference! By making clever use of how context vectors are integrated with LLM generation, REFRAG is able to make TTFT (Time-to-First-Token) 31X faster and TTIT (Time-to-Iterative-Token) 3X faster, overall improving LLM throughput by 7X! REFRAG is also able to process much longer input contexts than standard LLMs!

Most of the RAG systems today that are built with Vector Databases, such as Weaviate, throw away the associated vector with retrieved search results, only making use of the text content. REFRAG instead passes these vectors to the LLM, instead of the text content! This is further enhanced with a fine-grained chunk encoding strategy, and a 4-stage training algorithm that includes a selective chunk expansion policy trained with GRPO / PPO.

I hope you find the video useful! Happy to answer any questions, or discuss any ideas about REFRAG!

Chapters
0:00 REFRAG Explained!
1:58 REFRAG Architecture
5:20 Speed gains
8:50 Training Stages for REFRAG
12:15 RL for Selective Expansion
16:45 Experimental Results
21:32 Ablation Studies
24:55 Personal Takeaways

Links
REFRAG Paper Link: https://arxiv.org/abs/2509.01092
Transformers as Universal Computation Engines: https://arxiv.org/abs/2103.05247

----

Transcript:

[00:00:00] Hey everyone, thank you so much for watching this video explaining the refrag model from Meta Super Intelligence Labs. Refrag is super
[00:00:07] exciting for weave and vector databases on the whole, showing a new way to use precomputed vector embeddings with large
[00:00:12] language model inference, weaving together AI and database systems. Refrag speeds up time to first token by 31
[00:00:19] times. So imagine the difference in user experience if your chatbot responds in say 1 second compared to 31 seconds.
[00:00:25] Refrag is particularly effective at time to first token acceleration, but it also improves time to iterative token
[00:00:31] generation by three times and throughput by seven times. Now, this depends of course on how long the outputs you're
[00:00:37] generating are as well as a particular hyperparameters of refrag, but overall it's a huge unlock for LM inference
[00:00:43] speed as well as long context inputs. Refrag with uh K equals 32 can process
[00:00:48] 80 input documents, four times faster than a standard LMA 2 decoder could process 10 input documents. So this lets
[00:00:55] you give a massive amount of context to the large language model helping overcome this loss in the middle problem
[00:01:01] and fusing together advances in long context processing with large language models with retrieval augmented
[00:01:06] generation retrieved context and all this kind of stuff. So this is a complete gamecher for both LM inference
[00:01:13] speed as well as context lengths and feeding additional context to large language models. This video will explain
[00:01:18] the particular novelty behind how Refrag us uses vector databases to speed up LM inference as well as all the details
[00:01:24] behind the algorithm like the chunking strategies, the reinforced learning for selective expansion, the curriculum
[00:01:29] learning algorithms they apply, just diving into everything that's mentioned in the paper, algorithm experimental
[00:01:35] results, ablation studies as and concluding with some personal takeaways that I had from studying this paper and how I think it connects to different
[00:01:41] themes in AI and and just yeah, how this connects to that. So, thank you so much for watching. I really hope you find
[00:01:46] this useful. If so, please consider leaving a like on the video and subscribing to weate on YouTube.
[00:01:58] All right, let's dive into it. So, this image is showing the key innovation behind the refrag model architecture.
[00:02:03] Refrag is short for representation for rag. It's about a new way to encode the context tokens that you're adding to the
[00:02:10] input in retrieval augmented generation. So typically what you do in retrieval augmented generation is say you're
[00:02:16] retrieving context from something like a vector database and then you get the context back and you just throw away the vector. You don't use the vector
[00:02:22] anymore. You just take that input text, you smash it into a massive input text string that says say search results and
[00:02:29] then it's the 10 search results and that just gets put into the decoder transformer. But now what you're doing with refrag is you're making more use of
[00:02:36] the premputed vector representations from that context that you're adding into the input to the decoder
[00:02:41] transformer. So there's a bit of nuance to particularly how refrag does this. So this is probably the key detail to pay
[00:02:48] attention to and understand the sort of how we're how how particularly we're using these precomputed vectors. So we
[00:02:54] still have this concept of the chunk in the vector database and that might be something like say it's typically uh 500
[00:03:01] token chunks or say 300 token chunks and you still have that sized atomic unit in
[00:03:06] the vector database. But now you retrieve the say top 10 search results from the vector database with those say
[00:03:12] say there 256 token chunks in each of those uh search results that are retrieved from the vector database. Now
[00:03:18] what happens is you have an even more fine grained unit that is being processed with refrag. So for each of
[00:03:25] your 256 token chunks, say you now have eight 32 token chunks or say 16 16 token
[00:03:32] chunks and that's now going to be sort of like the second level of granularity behind what you're representing in your
[00:03:37] vector database and what's now being processed with refrag. So say you had this passage uh Donald Trump is the
[00:03:43] president of the United States. He assumed office on January 20th 2025 making blah blah blah. So you chunk this
[00:03:49] up into these even more fine grain units. So now you would have say the token vector for each of these uh for
[00:03:56] each of the tokens in each of the chunks and that's encoded with an embedding model that produces one vector for the
[00:04:02] 16 tokens. So now you've already compressed the input that the decoder is
[00:04:07] going to receive. We'll see some matrix multiplications in a second that'll make that make more sense. But you've compressed these even more fine grain
[00:04:13] units into just one vector instead of 16 token vectors. So now comes the really interesting thing and a super novel
[00:04:20] aspect of this. Now you're going to use a policy that's been trained with reinforcement learning using GRPO and PO
[00:04:26] to expand some of these compressed representations. So you've compressed Donald Trump is the president of the
[00:04:32] United States to one vector. But now this policy has decided that we actually need to represent each of the token
[00:04:38] vectors. So now you change that one vector for this entire chunk back into
[00:04:43] 16 token vectors. And then in the end you combine all these vectors. So you have one token vector for each of the uh
[00:04:49] tokens in the query. So the query is or the like the task input that's independent of the context that's
[00:04:55] retrieved in the rag sense is who is the president of the USA. You take these say seven tokens token vectors that are used
[00:05:02] with that. You add that with say the 16 token vectors that were expanded from the compressed representation of Donald
[00:05:08] Trump as the president of the United States. Then you add it with these other single vector compressed subunits of
[00:05:14] these other other parts of the context. Overall reducing the input size and making this way faster. So how much
[00:05:20] faster is it? Refrag is 31 times faster on time to first token than a standard
[00:05:25] llama 2 decoder transformer. So 31 times faster in time to first token. This is
[00:05:30] especially impactful when you have these online applications. Say you have streaming baked in. And overall this is
[00:05:36] really going to change how the user perceives the latency when you have 31 times faster time to first token. Say
[00:05:42] it's a chatbot or any kind of uh interaction where you have an online interaction between the user and the AI
[00:05:49] system. So you also have three times faster time to iterative token. This is once the transformer starts generating
[00:05:54] how quickly it can generate the next token and you have seven times faster overall throughput by using refrag. So
[00:06:00] already I think that sells the benefit of refrag and inspires the excitement in it is just way faster than these decoder
[00:06:06] only transformer models. Not only to also mention that it can process way longer context inputs. So to step a
[00:06:12] little into why this is faster without going too deep into the math, you basically have these two phases of LLM
[00:06:18] inference, the prefill phase and then decoding. So in prefill you have this N squared quadratic complexity thing where
[00:06:25] you compute the full attention matrix. Whereas in decoding you just take in the new token as it's auto reggressively
[00:06:31] decoding it one at a time. It's got this causal attention mask and so you're just adding on this you just have this one by
[00:06:37] D that then goes by the D by embedding dimension. So, so you just have this uh order n computation in decoding. So,
[00:06:43] that's why the the quick math looks like this where overall say the limit without the RL expansion is 16,000 tokens are
[00:06:51] then limited to 1,000 chunk embeddings. So, you already have an input reduction by about 16x. And again, the RL might
[00:06:57] expand some of these uh chunks. So, it's not that's like kind of the theoretical lower bound if you don't have any
[00:07:03] expansion. But then you're particularly going to see that time to first token advantage because instead of having
[00:07:10] 16,000 by 16,000 which is 256 million compared to 1,000 by 1,000. So you have 256 less you know flops floatingoint
[00:07:17] operations in prefill which is that n square thing. And then you also going to have faster decode time but not as much
[00:07:23] as that prefill attention. And then you also limit the size of the key value cache because you're reducing the latent
[00:07:29] dimensionality of this um of the input because you've shrunk the input. So you have a smaller key value cache. So
[00:07:35] that's kind of the quick math behind understanding why particularly it's so much faster for time to first token
[00:07:40] compared to time to iterative token. It's still faster, but just just to understand more so what's happening with
[00:07:45] this. Uh so in the paper they also I'm not going to explain this too much and dive into this but they also show
[00:07:50] similar similarly to what I just showed you but with more detail about all the different things that go into the
[00:07:56] latency bottlenecks, memory bottlenecks of transformer inference and how refra can help. Okay, so you're about halfway
[00:08:02] through understanding the refrag model architecture. To recap a little bit and tlddr it, uh the transformer decoder is
[00:08:08] processing compressed context tokens as input instead of the raw context tokens. As we saw, there's a little more nuance
[00:08:13] to particularly how it does this with these more fine grained uh context chunks used for refrag with this
[00:08:19] selective expansion policy train with uh reinforcement learning that may expand the context chunk back into the token
[00:08:26] vectors that were used for making up that chunk. So now onto the second part of the key details behind this. This is
[00:08:32] going to require a four-stage training algorithm to align the encoder and projection layer with the decoder. Train
[00:08:38] the selective expansion model with reinforcement learning and then fine-tune the end toend model for downstream tasks. I put just decoder on
[00:08:44] the slide, but it should say the whole system is trained end to end for downstream tasks. So all right, let's dive into the training. So training
[00:08:50] refrag begins with these four different stages that we'll be you know exploring.
[00:08:56] So we start off with reconstruction. So reconstruction is where the encoder and projection layer are trained to
[00:09:01] reconstruct the context tokens from their chunk embeddings while the decoder is frozen. This teaches the encoder to
[00:09:07] compress K tokens into a single embedding with minimal information loss and aligns the encoder output space with
[00:09:12] the decoder's token space. So as you remember from the architecture picture this is referring to training this
[00:09:18] encoder as well as a projection layer. So the idea behind this is that the encoder could be a pre-trained embedding
[00:09:24] model. In their particular case, they use the Roberta embedding model, but so you could use any embedding model. And
[00:09:31] so you're going to be fine-tuning the encoder as well as this projection layer. So this projection layer is a linear mapping from that uh the output
[00:09:38] space of the embedding model into the decoder's input space. So say the just purely in terms of understanding the
[00:09:43] mechanics of how you make the inputs work. Say your embedding model outputs a 768 dimension vector, but the
[00:09:49] transformer decoder is expecting 1,024. So you would have a projection layer
[00:09:54] that's just mapping from that 768 to 1,024. But then also the idea is to try
[00:10:00] to align the embedding spaces. This is I mean I think this is one of the most confusing concepts in in transformers.
[00:10:07] This there is this paper called uh transfer transformers as universal computation engines some title like
[00:10:12] that. This saying like transformers can attend over any arbitrary embedding space. But the idea of this is to align
[00:10:19] these embedding spaces and they do show in their ablations that this is critical to making it work. So that's the idea is
[00:10:24] that you're trying to push that encoder representation space the encoder that's compressing the vectors into the latent
[00:10:30] space of the decoder. So in the second space is a second stage of pre-training is continual pre-training. So then the
[00:10:37] decoder is unfrozen and the entire model is learning this next paragraph prediction task. The language modeling
[00:10:43] predict the next token task where the chunk embeddings from the first S tokens help predict the next O token. So
[00:10:48] another key detail to this is the use of curriculum learning. They're going to start by only predicting a single
[00:10:53] paragraph and then they're going to extend to predicting L chunks or you know increase the size of the outputs
[00:10:59] that it's predicting as it's doing this uh next token prediction end to end now with gradients going from decoder back
[00:11:05] to projection layer back to encoder training this whole thing end to end. So now step three you're bringing in the
[00:11:11] reinforced learning for selective expansion. So, a lightweight policy network, which we'll look at in a second, is learning which of the chunks
[00:11:17] should be expanded to full tokens versus staying compressed. And it's using the negative perplexity of the decoder
[00:11:22] predicting the remainder of the sequence as the reward signal for how well it did on this discrete optimization task of
[00:11:28] predicting which tokens to expand. So, the policy is taking the chunk embeddings as input and then is
[00:11:33] sequentially selecting which chunks to expand and that enables this dynamic compression rates. We'll see this in more detail in a second. So then stage
[00:11:40] four then you've trained this model decoder projection layer encoder and this reinforcement learning policy for
[00:11:46] expansion. You're training that end to end on downstream tasks like question answering multi-turn conversation and
[00:11:51] summarization to adapt the system. Now I'm not sure if you need to do this downstream task tuning but this is
[00:11:57] definitely something I've seen in these kind of Facebook paper this kind of like endtoend perspective. This was a big
[00:12:02] thing early on as you had models like uh railm or even the original rag paper does this. you have this end toend
[00:12:08] gradients from decoder going back to the encoder or embedding model and this is certainly an interesting way to think
[00:12:14] about these problems but anyways let's kick it off by this is I think the most unc of of these things a pretty standard
[00:12:20] I mean end to end training of a decoder and embedding model is certainly you know pretty advanced level stuff but I'd
[00:12:26] say this is probably the most novel aspect of this is training this uh selective expansion policy using
[00:12:32] reinforcement learning so the idea of selective expansion is that you have a policy network that is a neural network
[00:12:38] and this should be you know parameterized by theta but I didn't have that shortcut or I just got too lazy but
[00:12:43] anyway so you have this policy network that's taking the chunk embeddings as input so so you have these chunk
[00:12:49] embeddings it takes a uh say you had let's just say it's 16 even though it's going to be larger than that but you
[00:12:56] have 16 vectors and it's taking that say each of the vectors have 768 dimensions so it's taking that 16 by 70 768 as
[00:13:03] input and then it's predicting these discrete actions of which of the chunks to expand. So say it selects chunk one,
[00:13:10] then it selects chunk three, chunk eight, so on. That's the idea. And so you have this neural network that's
[00:13:16] predicting which chunk to expand. So say L of T. L is the chunk and T is like the time step. And so it's predicting it
[00:13:23] given the input X that 16 by 768. And then also given the chunks that it's
[00:13:28] selected so far. So so it's iteratively selecting chunks to expand. say so far it's already expanded chunks three,
[00:13:35] eight, four. It's receiving that in its input as it's predicting which one to expand next. And so that ends up being
[00:13:40] this uh you know softmax probability over the chunks that it's going to. So that's just how you sample from it. So
[00:13:46] say it put uh you know 60% probability on chunk six, 30% probability on chunk one. So you're sampling from that to
[00:13:54] predict which action to take next. And that's key to understanding how GRPO works. So GRPO the these are GRPO and
[00:14:01] PO. PO is kind of you know earlier paper 2017 more foundational in the space but
[00:14:06] these are these they both introduce these little tricks to how you make the loss functions work with reinforcement learning. So GRPO particularly you're
[00:14:13] trying to uh you're trying to calculate the advantage of the actions that you took and the idea of that is um how much
[00:14:19] better are these actions that I just took compared to the average actions or you know what I would or is the reward
[00:14:25] compared to what I would expect the reward to be given my current value uh network. So sorry I'm kind of confusing
[00:14:31] these topics, but usually what you do in reinforcement learning is you have these uh you have like the Q network and then
[00:14:36] you have the value network. So you have the action model as well as the actor critic that kind of way think about it.
[00:14:42] So usually you have the actor which is your policy network and then you also have some kind of value network and now
[00:14:47] you're training these two neural networks at the same time. So anyone who trained GANs or been down that rabbit hole knows that's kind of a pain. So
[00:14:54] GRPO is instead using uh the group mean. So instead of having a value network that calculates the value of this state
[00:15:00] action like what you expected the value to be compared to what you got, you're going to be estimating it from the group. So the idea of the group is uh
[00:15:07] you know we sample a ton of different chunk expansions and then we get the reward which is a negative perplexity
[00:15:13] from our decoder from all those different potential chunk expansions and that becomes like the group mean the group standard deviation. So this is how
[00:15:20] you get that kind of adjusted advantage. Uh so so then the idea from uh PO that we're going to be using is u and there's
[00:15:28] a little more to PO but this particular uh clipping idea is what's kind of like extracted and then used in sort of our
[00:15:33] reinforcement learning recipe for this. Okay. So the the TLDDR of the idea is
[00:15:39] that you are uh if if this action sequence from the new policy is like way
[00:15:45] higher than the old policy then you're just going to sort of you know clip the loss. you're not going to let it take
[00:15:51] such a huge gradient step in that direction. So, it's a way of controlling. You don't want to have these massive gradient steps that say
[00:15:58] take you, you know, out of take you into some region that's off the landscape and
[00:16:03] just generally control how much you update the network at each stage. That's the TLDDR idea of it. So, basically, you
[00:16:09] have this ratio, the ratio of the new policy compared to the old policy and how likely this action is. And you're
[00:16:15] going to do this for all the actions. So, you know, you're going through one to t prime. prime is the the length of
[00:16:20] how many chunks you chose to expand. So say it's four like chunk 1 3 5 11 even
[00:16:26] though you could do it in any order. It could be 12 3 4 1. Anyways, so you have this probability of this action in the
[00:16:32] new policy compared to the old policy. You multiply that by the advantage and you're going to clip it with this
[00:16:38] epsilon which is usually like you know 0.8 or 1.2. So you have that way of clipping the the update. So now let's
[00:16:45] dive into the experimental results of the paper. Earlier on they showed that refrag is way faster and it's obviously
[00:16:50] exciting for that reason. But then in the latter end does it still retain the performance the are the generations of
[00:16:55] refrag just as high quality as normal large language models. And so to begin before we dive into the results let's
[00:17:01] look at what they're comparing refrag against. They start off with two different variants of the llama 27
[00:17:07] billion parameter model. Llama full context and llama no context. So llama no context is sort of like just like a
[00:17:13] sanity check. It's where you have none of the input context to predict the next token. So maybe what's interesting about
[00:17:20] that is you're this is measuring perplexity on different output token sequences. And so with no context is
[00:17:27] kind of creating the context as it's decoding. So that that's maybe something interesting about the no context case.
[00:17:32] But generally this is just sort of like a you know a baseline to just see where is it at without any input context. So
[00:17:38] then llama full context it's seeing the full context and then again it's across these pretty small you know I I guess
[00:17:45] small for my taste where we're at currently today but anyway so 5124
[00:17:50] 2048 these are the different amounts of context that that these particular plots are showing as you see with the p 512 so
[00:17:56] on so llama full context is seeing that entire 512 context before it's predicting the next token llama 32k they
[00:18:03] describe this is a llama 27 billion parameter model that's been fine-tuned to process 32,000 input lengths and then
[00:18:09] you have llama 256 is a rolling window of the last 256 context tokens and then
[00:18:15] you have two other prior works that I'm not familiar with myself replug and c
[00:18:20] and then you have k the k is for the different parameterizations of refrag so when refrag is refrag 8 that means those
[00:18:27] those subcontext units are eight tokens 32 32 tokens and so on so so the
[00:18:33] important thing is showing that refrag is retaining the language model perplexity across these different uh
[00:18:39] output token sizes compared to the original llama model showing that not only is it faster but it's also retaining the performance and again they
[00:18:46] show this on even longer context length inputs. So in this case because this model uh isn't isn't trained to process
[00:18:53] these really long inputs. It's uh failing more so on these really long contexts. All that kind of you know uh
[00:19:00] lost in the middle all that kind of stuff is coming in when you're trying to give the original llama 2 7 billion parameter transformer model say a 8,000
[00:19:08] context input to predict the next token. So then looking at the downstream application results. Okay great. It can
[00:19:15] predict the next token. Can it answer my questions? Can it chat with me? And all this kind of stuff. So uh again maybe
[00:19:21] before even diving into it this plot is kind of showing you a couple things. So firstly it's showing you the uh
[00:19:27] comparative amount of passages that refrag can read compared to llama fine tetunes. This is the llama fine tune
[00:19:32] model but because refrag is able to uh compress the passages with refrag 8 and
[00:19:39] eight context passages. You're seeing the same amount of tokens processed as llama fine tune with one passage. And
[00:19:45] then when you have uh when you increase refrag to 32. So 32 means because you are have uh 32 tokens in the compressed
[00:19:52] vector you're overall compressing the input much more than say if there was eight tok if you have eight tokens per vector then you have four chunk vectors
[00:20:00] compared to 32 making it one chunk vector. So you then have four times less tokens. So it's going to be four times
[00:20:05] faster. So that means refrag 32 with eight passages which which you see is
[00:20:10] retaining the performance on these questionans answering tasks is going to be four times faster than llama fine-tune with just one passage. So this
[00:20:18] is especially pronounced when you then move into the long context case and you're talking about 10 passages compared to 80 passages. So refrag it's
[00:20:26] solving this long context problem. It's able to take in massive context inputs,
[00:20:31] 80 passages, and retain the uh performance of llama fine tune with 10 passages while also being way faster
[00:20:38] with refrag 32. And then they're further showing the gains when you have uh multiple choice tasks and overall just
[00:20:45] yeah, so just overall getting a sense of refrag is retaining the performance while being able to process much longer
[00:20:52] uh context inputs and then it's also way faster. So, uh, they they do another
[00:20:58] ablation where they're comparing using a strong retriever with a weak retriever. Maybe I have some thoughts to share on
[00:21:03] that and some personal takeaways at the end. Uh, and then they're also showing multi-turn rag. I'm not super familiar
[00:21:08] with these data sets, but they have the there's also benchmarks for conversational search and conversational
[00:21:14] question answering where is the the challenge is now you have these long conversations. So, just finding the
[00:21:20] search query, retaining the context, referencing old chats, it's a very interesting long context problem. And so
[00:21:25] again, they're just ablating the number of passages that are used when you're retrieving in these different conversational uh questionans answering
[00:21:31] tasks. The authors also present a few ablation studies showing the importance of different details of the refrag model
[00:21:36] architecture and the training algorithm. So starting off, they show that the curriculum learning is essential for
[00:21:41] reconstruction. So when they start off with this uh they start off by freezing the decoder and then aligning the
[00:21:47] encoder and projection layer and then they unfreeze the decoder and continue pre-training by to just I think get the
[00:21:53] decoder used to processing that embedding space and that kind of idea. Uh so they show that using this
[00:21:58] curriculum of starting off by just predicting one paragraph as a at a time compared to then say or at one chunk and
[00:22:05] then L chunks having that curriculum strategy is a huge part of making this work and the sort of training recipe for
[00:22:10] refrag. Um and then they show that the reconstruction phase is essential. So aligning the encoder space with the
[00:22:16] decoder space they show that this is a critical part of making this work similarly as the uh curriculum learning.
[00:22:23] So you know just showing the importance of the different details of the training recipe. So then this part is something that I think is interesting is showing
[00:22:29] this the advantages of the reinforce learning based selective compression. So they're comparing reinforce and learning
[00:22:34] the you know the policy expansion the the policy train to expand the chunks that we saw compared to say using the
[00:22:40] perplexity of the chunk to expand it. So say you have a you know a chunk that has
[00:22:45] high as the language model is predicting the next token in that chunk it has really you know high entropy of the
[00:22:51] predicting the next token you that would be a good chunk to expand. It's a good signal to that that that needs more fine
[00:22:57] grain detail. It needs an embedding per token in that chunk. And in my opinion, they're showing pretty similar. And then
[00:23:02] and then random as a, you know, just randomly selecting chunks to expand. So I think they're showing a pretty similar
[00:23:08] perplexity for each of these strategies. So I'm not sure I'm not sure if it's worth the extra effort because that
[00:23:14] reinforce learning for chunk expansion, that's obviously not super easy to train something like that. So I'm not sure if
[00:23:19] this graph is super convincing that you need to do that reinforcement learning for chunk expansion phase. So then
[00:23:25] they're also showing what what if we just don't expand any of the chunks. And so they're comparing refrag and and then
[00:23:32] they're comparing refrag 8 with 16 plus the chunk expansion, but anyway. So so you're seeing the difference in the
[00:23:38] performance. It doesn't look to be uh that significant to me. Uh so then they're also showing the difference in
[00:23:43] the training curves of the different compression rates. So this is a huge detail of refrag. The the higher you set that k when you have like refrag sub 16
[00:23:50] 32 whatever the the higher it is the more you're compressing the input length. So they're showing the training curves compared to refrag 8 16 32 64. So
[00:23:58] it looks like it's easier to train the higher compression, right? Because you have more gradient signal. But then I
[00:24:03] guess they're showing you but then the higher the compression, the more you can scale it. So I think probably there's
[00:24:08] still a lot of variables to this one. Uh and then they're showing this chart again. This is from earlier just to kind
[00:24:14] of explain this concept a little further where you have refrag 8 1632 and it seems like pretty similar results with
[00:24:21] it seems like the results are a little better here with the uh with with the with less compression
[00:24:27] and then maybe the results are better a bit here with with when you have longer inputs more compression so I think that's definitely an interesting
[00:24:32] question uh and then showing uh different encoder and decoder models so exploring the Roberta base or Roberta
[00:24:39] large embedding model and then showing the llama 27B B versus Lama 2 13B decoder model and it looks like 13B is
[00:24:46] easier to train. So that's definitely interesting. I mean as you'd expect larger model generally more capable and
[00:24:51] so anyway so just understanding the impact of the different embedding models. Awesome. So hopefully from that overview you have a good sense of the
[00:24:57] the mechanics of refrag such as how it uses this chunk compression this concept
[00:25:02] of chunk expansion and then understanding how this reduces the input length for the decoder transformer how
[00:25:08] that then reduces say the prefill attention decode attention and then understanding this four- stage training
[00:25:13] process to align the encoder and projection layer with the decoder's latent space have the train this RL for
[00:25:19] selective expansion policy and see some of the results they're presenting as well as the ablations I wanted to end this video with some personal takeaways,
[00:25:25] some reactions that I had to reading this refrag paper and learning more about this algorithm. So firstly, I
[00:25:31] think the high level thing is this long context and retrieval augmented generation. So I think you know as the
[00:25:37] story of this has been early on rag was so popular because LMS had a super small
[00:25:43] context window. They could only process say 4,000 tokens and then 8,000 tokens but still pretty limited. And so this
[00:25:49] idea of retrieving relevant context for the inference from the vector database or the you know database search engine
[00:25:55] this was just so impactful but I I don't understand why long context and retrieval augmented generation don't
[00:26:02] complement each other really well. Say you know already with information retrieval we're measuring this recall at
[00:26:08] K. And the higher we increase K from 10 to 20 to 100 the easier it is to find the gold document. So if we're able to
[00:26:15] set K to 100 and we can give you the top 100 search results and we solve this loss in the middle problem, figure out
[00:26:20] ways to compress the context, it just makes the information retrieval task a
[00:26:25] lot easier. But then probably more interestingly than that is this concept of diversity in search results. So
[00:26:30] rather than just giving you our guess at the top 100 most relevant search results, if we can instead have some way
[00:26:36] of saying these these documents cover this topic, these documents cover this topic, we give this diverse context, I
[00:26:43] think that's just such a more exciting direction for retrieval augmented generation. So I really enjoyed learning
[00:26:48] about the fresh stack benchmark from Nandon at all spoken with Nanon on the WVA podcast and this is a benchmark
[00:26:54] that's measuring nugget coverage. So it's sourced from stack overflow. You ask them a question like how do I use
[00:27:00] lang chain and weave and the the answers are annotated by covering these different nuggets of knowledge that you
[00:27:05] need to answer these questions. I think this is going to be a huge part of the future of information retrieval benchmarking as well as say bright and
[00:27:11] all this kind of stuff. So we something we're doing to help with this is we have this multicolction retrieval. So this is
[00:27:17] one way that when when you're using the query agent search mode like this and you're searching papers, blogs, notes or
[00:27:23] say a lot of people, you know, sort of the rag is dead group I think mostly comes from people who are uh using rag
[00:27:28] for coding assistance and so they search their code mostly and they're saying oh do because it's easier to just look up
[00:27:34] the function but say you're retrieving from your code as well as your blog posts as well as say notes papers and
[00:27:40] you just give the agent so much context from these different source. I think this is one great way to have diversity
[00:27:45] as well as also the save the query agent can use filters in your data and I think this is another really interesting way
[00:27:50] to have a navigate and get that uh get that diversity in the search results as well as I think the vision behind fresh
[00:27:56] stack and the way that nonun sees is just my perspective of how PCs it is that there would be ways to get this
[00:28:02] diversity directly in the vector embedding space and there are things like say maximal marginal relevance or I
[00:28:09] don't know if I got the M's backwards there but you can do things in vector space say vector clustering. You could also just get diversity directly in the
[00:28:16] vector space as well as I think using symbolic filters. So I think a lot of interesting things there with the
[00:28:21] broader topic of not just feeding the LLM, you know, a ranked list of most
[00:28:26] relevant to your query, but also having some diversity in these search results. And I think that'll be really exciting
[00:28:31] as we have longer context LMS getting better at processing a longer context and then better at using the retrieved
[00:28:37] context. So next thing to talk a little more about refrag with weave. I think
[00:28:42] this this waves of vector database usage patterns is super fascinating. We started off by scaling the single
[00:28:48] collection a billion vectors in the same collection. Uh we had earlier blog post say from Bob back in I think 2022 about
[00:28:56] putting sphere in WV8 and this was like you know putting the internet and we was the idea we had a billion vectors in a
[00:29:02] single collection. Then as we started to you know as WVA grew and had more customers this multi-tenant pattern
[00:29:07] became really impactful where you say you have the same schema but then there's a million tenants per schema and
[00:29:13] then you have this kind of data isolation scaling these tenants and that was sort of the next vector database
[00:29:19] usage pattern but now I think we're coming into the third pattern which is for whereas in the early days of we
[00:29:25] there was one object one object had one vector now we have tons of vectors for each of the objects so I think currently
[00:29:31] with refrag what you see is uh say you have 256 tokens per chunk. You'd have one vector for that and then and then
[00:29:38] let's say you're just searching with HNSW snowflake embedding is sort of standard setup. You'd have the one
[00:29:43] vector for that. Then you'd have the 16 vectors for the the subcontent units and then you'd have the 256 token vectors.
[00:29:51] So 273 vectors to represent that one data object. And now so so and then
[00:29:56] that's searching with a single vector. But you might also say search with Colbert late interaction. So, so you
[00:30:02] might have represented the entire content with say 256 token vectors for that reason and then you have the same
[00:30:09] and then you have these subunits to then feed into refrag but generally this pattern of you know mult multiple
[00:30:15] vectors per object this is a huge change I think in how people are using vector databases so I'm really excited about
[00:30:21] WVA has named vectors u colbert late interaction support things like move vera and then all this quantization
[00:30:28] research has gone is really impressive this rotational quant ization. I highly recommend checking this out. And there's
[00:30:33] ways that we can compress vectors as we're storing tons of vectors. So, another uh podcast I really liked was
[00:30:39] with Bob and Ben from Ben Cus from Box AI. And at the time, I hadn't understood this well, but they were discussing a
[00:30:46] lot about um we already have pabytes of data, tons of data. What happens now that all objects come with a vector
[00:30:53] representation or tons of vector representation? So this podcast started putting my thinking into this space of
[00:31:00] representing objects with tons of vectors. So maybe something you'll find interesting. Definitely something that helped me, you know, is is biasing my
[00:31:07] perspective on thinking that we're headed towards this world where you represent objects with tons of vectors. Awesome. So another takeaway kind of
[00:31:14] this theme of um merging model inference with vector databases.
[00:31:19] I think this many shot in context learning thing can maybe be more democratized with this refrag model architecture. So many shot in cog text
[00:31:27] learning it's mostly been pioneered by Google and they are killing it at these long context LLMs like Gemini and you
[00:31:33] know the different variants of Gemini and so this particular paper they're showing that uh so with LM training we
[00:31:39] have in context learning where you show examples of inputs and outputs and that teaches the LM how to do the task and
[00:31:45] what they're showing with many many shot in context learning with their long context models is that you could say
[00:31:50] give 500 examples of the task in the input And that's resulting in these big gains
[00:31:57] over say fourshot examples. So I think refrag could be a way to compress the
[00:32:03] context with these many shot in context learning examples and overall make this
[00:32:08] something that's uh a more popular way of how you would fine-tune your language model is by using incontext learning
[00:32:15] with this sort of scalable algorithm for doing so. Uh so then lastly this one is more of a reach. This is I've been
[00:32:21] looking into these kind of information retrieval systems and one paper that's really fascinating is um optimizing compound retrieval systems. So the idea
[00:32:28] behind uh optimizing compound retrieval systems is uh when you have reranking
[00:32:34] you have these different kinds of reankers. You have pointwise reankers you have pair-wise reankers you have
[00:32:40] these set wise reankers and you have listwise rankers where the idea is uh which documents are you compare are you
[00:32:46] pointwise like cross encoders take as input a candidate document and the query as input. So there's no relative
[00:32:51] comparison. Whereas say a listwise ranker takes in all the candidate documents in the query. So you have this
[00:32:56] policy that's selecting which documents to compare against each other. So there's another kind of case of using
[00:33:02] this like uh discrete policy on top of so in the case of optimizing compound
[00:33:07] retrieval systems the policy is choosing which of these documents should be put into a pair-wise comparison and then
[00:33:13] which only need the pointwise score. So pretty interesting algorithm but generally I just just this is something I'm thinking about and this kind of
[00:33:19] alignment of using these RL layers on top to these policies in the middle that do discrete actions I think it's quite
[00:33:26] interesting even though as we saw in the inflations of the paper I'm not yet completely convinced that uh that this
[00:33:31] is necessary for reffrag and then finally what if we had a ga optimized context expansion so in the reffrag
[00:33:38] paper they're doing this discrete action uh policy you know it's just selecting
[00:33:43] the chunks based on the chunk embeddings. But what if instead you use the GAP idea of natural language where
[00:33:48] it's seeing the text of each of the content chunks and then it's using natural language to decide which ones to
[00:33:55] expand and then the perplexity metric is somehow gives you natural language feedback back to some maybe explanation
[00:34:01] of why that content chunk should have been expanded. So generally and then I generally think it would be a little bit
[00:34:06] easier to use GEA than GR than that GRPO PO mix that we just saw. So I think it'd
[00:34:12] be very interesting to see what refrag looks like with in a natural language space with optimized gapa. Thank you so
[00:34:17] much for watching this video diving into the refrag model from meta super intelligence labs. If you found this
[00:34:22] video helpful, please consider liking the video and subscribing to we on YouTube. I'd also love to know what you think about refrag and all these
[00:34:28] different ideas. So please leave a comment if you like us to answer any questions or discuss any ideas you have about refrag. Thanks again for watching.
[00:34:35] [Music]