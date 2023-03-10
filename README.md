# Federated Intrusion Detection In Autonomous Transportation Systems

The increasing number of automation in the transport sector is accompanied by a big problem:
A raising number of cyberattacks all over the world and the highest possibility to be exposed
to tremendously harmful and cost-intensive attacks ever. Intrusion Detection Systems are one
of the most essential parts of cybersecurity and a highly effective mechanism to fight the grow-
ing threat. However, they have several problems in real applications, where the needed data
is heterogeneous and spread across multiple locations, and cannot be shared due to regulatory,
competitiveness, or privacy reasons. Combined with one of the latest concepts in machine
learning, the so-called Federated Learning, we could reach a new era of Intrusion Detection
Systems. Federated Learning enables multiple parties to collaboratively construct a machine-
learning model and improve the performance. We propose a framework using the popular Fed-
erated Averaging algorithm for the aggregation of autoencoder weights for intrusion detection
in modern autonomous transportation systems. We present the customizable machine learning
model, the federated optimization algorithm, and our own communication protocol with en-
crypted data transmission. In extensive experiments on the CIC-IDS17 dataset, our algorithm
achieves a high accuracy of around 90% in normal traffic, a detection time of less than 200ms
on average for anomalies, and a data throughput of around 150 samples per second. We showed
that Federated Learning allows advancements of up to 20% in True Positive Rate for individual
participating clients over a non-federated training setting. Implemented in autonomous trans-
portation systems, our system would ensure that the vehicle is always one step ahead of the
attacker.
