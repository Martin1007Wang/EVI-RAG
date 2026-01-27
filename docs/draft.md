## Methodology

In this section, we present the Query-Conditioned Bilinear Alignment (QC-BiA) framework for Knowledge Graph Question Answering (KGQA). We first formulate the problem as a Generative Flow Network (GFlowNet) task (Sec. 3.1). We then introduce our theoretical backbone based on Doob’s $h$-transform, which motivates a dual-stream architecture (Sec. 3.2). Subsequently, we detail the QC-BiA parameterization that enforces geometric inductive biases via dynamic feature modulation (Sec. 3.3). Finally, we describe the training protocol based on Detailed Balance with boundary anchoring (Sec. 3.4).

### 3.1 Problem Formulation

Let $\mathcal{G} = (\mathcal{V}, \mathcal{R}, \mathcal{E})$ denote a knowledge graph, where $\mathcal{V}$ is the set of entities and $\mathcal{E}$ contains directed edges $(u, r, v)$ with $r \in \mathcal{R}$. Given a natural language question $q$ and a starting entity $s_{start}$, the task is to identify an answer set $A \subset \mathcal{V}$ that answers $q$.

We model the reasoning process as a sequential decision problem. An agent starts at $s_0 = s_{start}$ and generates a trajectory $\tau = (s_0 \xrightarrow{r_0} s_1 \dots \xrightarrow{r_{k-1}} s_k)$ of length $k \le K$. The goal of GFlowNet is to learn a stochastic policy $\pi_F(\tau | q)$ such that the marginal probability of terminating at state $x$ is proportional to a reward function $R(x|q)$, i.e., $P_T(x) \propto R(x|q)$. In KGQA, $R(x|q)$ is typically a sparse binary signal indicating whether $x$ is the correct answer.

### 3.2 Theoretical Framework: H-Transform Guided Flow

Navigating large-scale KGs with sparse rewards poses a significant exploration challenge. We address this via the lens of Doob’s $h$-transform, which provides a theoretical mechanism to condition a random walk to hit a target subset.

Let $P_{ref}$ be a reference random walk on $\mathcal{G}$. The optimal policy $\pi^*$ to reach a target subset $A$ is given by twisting $P_{ref}$ with a harmonic function $h(u)$:

$$\pi^*(v \mid u) \propto P_{ref}(v \mid u) h(v)$$

where $h(u)$ represents the probability (or potential) of reaching $A$ from $u$.

In our framework, we explicitly model this potential using a Backward Teacher. We maintain two coupled flows:

* Forward Student ($P_F, F_F$): Learns the exploration policy $\pi_F$ conditioned on the question $q$.

* Backward Teacher ($P_B, F_B$): Approximates the optimal potential $h(u)$. Crucially, unlike prior works that use static backward policies, we model the teacher as a target-conditioned process. The backward flow $F_B(u \mid q, A)$ serves as a dynamic potential landscape, guiding the student towards high-reward regions.

### 3.3 Architecture: Query-Conditioned Bilinear Alignment (QC-BiA)

To bridge the gap between logical reasoning and graph representation learning, we propose the QC-BiA module. This architecture incorporates geometric inductive biases through Feature-wise Linear Modulation (FiLM), conditioned on specific agent contexts.

#### 3.3.1 Asymmetric Context Formulation

A core innovation of our framework is the strict separation of information visibility between the Forward Student and the Backward Teacher. We explicitly construct context vectors $\mathbf{c}$ to enforce the H-transform constraints.

Let $\mathbf{h}_q$ be the semantic embedding of question $q$ (e.g., from RoBERTa), $\mathbf{h}_{start}$ be the graph embedding of the topic entity $s_{start}$, and $\mathbf{h}_A$ be the pooled embedding of the answer set $A$. We use query-aware attention pooling:
$$
\alpha_i = \text{softmax}\left(\frac{\mathbf{h}_q^\top \mathbf{h}_{a_i}}{\sqrt{d}}\right), \quad
\mathbf{h}_A = \sum_i \alpha_i \mathbf{h}_{a_i}.
$$

* Forward Context (Student View): The student must reason based solely on the question and the starting point. We fuse the question semantics with the topic entity anchor:

    $$\mathbf{c}_{fwd} = \text{MLP}_{fwd}([\mathbf{h}_q \,;\, \mathbf{h}_{start}])$$

    Including $\mathbf{h}_{start}$ serves as a global structural anchor, ensuring the reasoning process remains grounded relative to the topic entity throughout the multi-hop trajectory.

* Backward Context (Teacher View): The teacher operates with "privileged information" (God view) to approximate the optimal potential. It conditions on the full triplet:

    $$\mathbf{c}_{bwd} = \text{MLP}_{bwd}([\mathbf{h}_q \,;\, \mathbf{h}_{start} \,;\, \mathbf{h}_{A}])$$

    This augmented context allows the teacher to effectively "see" the destination set, thereby generating a valid gradient signal that pulls the student towards the answers.

#### 3.3.2 Dynamic Relation Morphing

We define a unified scoring mechanism utilized by both policies. Given a context $\mathbf{c}$ (either $\mathbf{c}_{fwd}$ or $\mathbf{c}_{bwd}$), current node $u$, and relation $r$:

1. Projection: Map inputs to a latent space $\mathbb{R}^d$:

    $$\mathbf{z}_u = \text{LN}(\mathbf{W}_u \mathbf{h}_u), \quad \mathbf{z}_r = \mathbf{W}_r \mathbf{e}_r, \quad \mathbf{z}_c = \text{LN}(\mathbf{c})$$

2. Modulation: Generate affine parameters $[\boldsymbol{\gamma}; \boldsymbol{\beta}]$ from $\mathbf{z}_c$ to dynamically morph the relation space:
    
    $$[\boldsymbol{\gamma}; \boldsymbol{\beta}] = \mathbf{W}_{gate} (\sigma(\mathbf{W}_{inter} \mathbf{z}_c))$$
    
    $$\tilde{\mathbf{z}}_r = (1 + \boldsymbol{\gamma}) \odot \mathbf{z}_r + \boldsymbol{\beta}$$

3. Scoring: Compute the transition logit via bilinear alignment:
    
    $$\text{Logit}(v \mid u, r, \mathbf{c}) = \frac{(\mathbf{z}_u \odot \tilde{\mathbf{z}}_r)^\top \mathbf{z}_v}{\sqrt{d}}$$

### 3.4 Training via Detailed Balance

We optimize the network parameters using the Detailed Balance (DB) objective with a Student-Driven Trajectory Learning scheme.

#### 3.4.1 Student-Driven Sampling

Unlike prior works that attempt independent backward sampling (which suffers from the "vanishing overlap" problem in large graphs), we treat the Backward Teacher purely as a critic.

1. Rollout: The Student policy $\pi_F(\cdot | \mathbf{c}_{fwd})$ samples a batch of trajectories $\tau = (s_0 \to \dots \to s_k)$ starting from $s_{start}$.

2. Evaluation: For every transition $(u \xrightarrow{r} v)$ in $\tau$:

    * The Student predicts the forward probability $P_F(v|u, \mathbf{c}_{fwd})$ and the state flow $\log Z(u | \mathbf{c}_{fwd})$.
    
    * The Teacher evaluates the reverse transition $(v \xrightarrow{r^{-1}} u)$ using the privileged context $\mathbf{c}_{bwd}$. Crucially, the teacher answers: "Given we must end in $A$, how likely is it to step back from $v$ to $u$?"

#### 3.4.2 The Loss Function

The loss minimizes the flow mismatch error:

$$\mathcal{L}(\theta) = \mathbb{E}_{\tau \sim \pi_F} \left[ \sum_{(u, v) \in \tau} \left( \log Z(u|\mathbf{c}_{fwd}) + \log P_F(v|u, \mathbf{c}_{fwd}) - \log Z(v|\mathbf{c}_{fwd}) - \log P_B(u|v, \mathbf{c}_{bwd}) \right)^2 \right]$$

#### 3.4.3 Boundary Anchoring with Reward Shaping

To ground the flow values:

1. Success: If $v \in A$, we fix $\log Z(v) = 0$ (Reward $R=1$).

2. Failure: If a trajectory ends at $v \notin A$, we assign a penalty $\log Z(v) = C_{penalty}$ (e.g., $-10$). This "soft energy clamping" prevents the flow from exploding in dead-end regions and provides a negative gradient to suppress incorrect paths.
