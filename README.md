# Çoklu Ajan Pekiştirmeli Öğrenme & Oyun Teorisi: 52 Haftalık Medium Yazı Serisi

## 📖 Proje Hakkında
Bu repo, Multi-Agent Reinforcement Learning (MARL) ve Oyun Teorisi konularını derinlemesine inceleyen 52 haftalık Medium yazı serisinin resmi kaynağıdır. Seri, teorik temelleri, pratik uygulamaları ve son teknoloji algitmaları kapsayan kapsamlı bir eğitim yolculuğu sunar.

---

## 🎯 Serinin Amacı

*   **Teorik Derinlik:** Oyun teorisinin matematiksel temellerini MARL ile harmanlayarak sağlam bir kavramsal çerçeve oluşturmak.
*   **Pratik Uygulamalar:** AlphaGo, Pluribus, OpenAI Five gibi gerçek dünya başarılarının arkasındaki mekaniği açıklamak.
*   **Kod ile Öğrenme:** Her kavramı PyTorch ve TensorFlow ile çalışan örneklerle pekiştirmek.
*   **Araştırma Köprüsü:** Akademik literatür ile endüstri uygulamaları arasında köprü kurmak.

---

## 👥 Hedef Kitle

*   **ML/RL Araştırmacıları:** Çoklu ajan sistemlerinde çalışan veya ilgilenen akademisyenler.
*   **Oyun Geliştiriciler:** NPC AI ve prosedürel içerik üretimi için MARL kullananlar.
*   **Robotik Mühendisleri:** Swarm robotics ve multi-robot koordinasyonu alanında çalışanlar.
*   **Finans/Ekonomi Uzmanları:** Algoritmik trading ve piyasa simülasyonu ile ilgilenenler.
*   **Otonom Sistem Geliştiricileri:** Araç, drone filoları ve dağıtık sistemler üzerinde çalışanlar.

---

## Ön Koşullar

✅ Temel pekiştirmeli öğrenme bilgisi (Q-Learning, Policy Gradients)  

✅ Python programlama (NumPy, PyTorch/TensorFlow)  

✅ Temel lineer cebir ve olasılık teorisi  

⚠️ Oyun teorisi bilgisi gerekli değil (sıfırdan anlatılacak)  


---

## 📚 Serinin Yapısı
Seri 4 çeyrek halinde organize edilmiştir, her biri farklı bir temaya odaklanır:

### 🌱 Çeyrek 1: Temeller ve Nash Dengesi (Hafta 1-13)
Nash dengesi, matris oyunları ve temel denge konseptleri.
*   **Odak Konular:**
    *   Nash Dengesi'nin MARL'a entegrasyonu
    *   Mahkum İkilemi, Rock-Paper-Scissors gibi klasik oyunlar
    *   Mixed strategies, Stackelberg dengesi, best response dynamics
*   **Anahtar Algoritmalar:** Nash-Q Learning, CFR, Fictitious Play

### 🤝 Çeyrek 2: İşbirliği ve Koalisyon (Hafta 14-26)
Kooperatif oyun teorisi, sosyal dilemmalar ve evrimsel dinamikler.
*   **Odak Konular:**
    *   Shapley Value ile credit assignment
    *   QMIX/VDN arkasındaki matematik
    *   Public goods games, trust ve reputation
    *   Population-based training ve evolutionary stable strategies
*   **Anahtar Algoritmalar:** COMA, QMIX, VDN, Population-Based Training

### 🔀 Çeyrek 3: Asimetri ve Kompleks Dinamikler (Hafta 27-39)
Bayesian oyunlar, network effects ve stokastik sistemler.
*   **Odak Konular:**
    *   Incomplete information ve mechanism design
    *   Graph Neural Networks + MARL
    *   Congestion games, epidemic models
    *   Markov games, differential games
    *   Risk-sensitive equilibria
*   **Anahtar Algoritmalar:** VCG Mechanisms, GNN-MARL, Hamilton-Jacobi-Isaacs

### 🚀 Çeyrek 4: Meta-Learning ve Frontier Topics (Hafta 40-52)
Öğrenme hakkında öğrenme ve gelecek araştırma yönleri.
*   **Odak Konular:**
    *   Meta-game theory ve opponent modeling
    *   Transfer learning across games
    *   Mean field multi-type games
    *   Quantum game theory
    *   Causal inference in MARL
    *   Bounded rationality ve Level-k reasoning
*   **Anahtar Algoritmalar:** MAML, Theory of Mind Networks, Hypernetworks

---

## 📋 Haftalık İçerik Listesi

### **Çeyrek 1: Temeller ve Nash Dengesi (Hafta 1-13)**

**Hafta 1-2: Giriş ve Motivasyon**
- **Hafta 1:** Neden Tek Ajan Yetmiyor? MARL ve Oyun Teorisinin Doğuşu
- **Hafta 2:** AlphaGo'dan StarCraft'a: Strateji Evriminin Anatomisi

**Hafta 3-6: Nash Dengesi ve MARL**
- **Hafta 3:** Nash Dengesi'ni Öğrenmek: Q-Learning Buluşuyor von Neumann ile
- **Hafta 4:** Çok Oyunculu Poker'de Denge Arayışı: CFR'ın Sırları
- **Hafta 5:** İşbirlikçi Nash Dengeleri: Mean Field MARL
- **Hafta 6:** Correlated Equilibrium vs Nash

**Hafta 7-10: Matris Oyunları**
- **Hafta 7:** Mahkum İkilemi'nden Kaçış
- **Hafta 8:** Rock-Paper-Scissors'ı Kazanmak
- **Hafta 9:** Mixed Strategy Equilibria
- **Hafta 10:** Zero-Sum'dan Non-Zero Sum'a

**Hafta 11-13: İleri Denge Konseptleri**
- **Hafta 11:** Stackelberg Dengesi ile Liderlik
- **Hafta 12:** Epsilon-Nash ve Approximate Equilibria
- **Hafta 13:** Best Response Dynamics

### **Çeyrek 2: İşbirliği ve Koalisyon (Hafta 14-26)**

**Hafta 14-17: Kooperatif Oyun Teorisi**
- **Hafta 14:** Shapley Value ile Adil Ödül Dağıtımı
- **Hafta 15:** Koalisyon Oluşturmanın Algoritması
- **Hafta 16:** QMIX ve VDN'nin Arkasındaki Oyun Teorisi
- **Hafta 17:** Communication Games

**Hafta 18-21: Sosyal Dilemmalar**
- **Hafta 18:** Public Goods Game'de MARL Ajanları
- **Hafta 19:** Güven İnşasının Algoritması
- **Hafta 20:** Altruism mu Egoism mu?
- **Hafta 21:** Reciprocity ve Fairness

**Hafta 22-26: Adaptif Sistemler**
- **Hafta 22:** Population-Based Training Meets Evolutionary Game Theory
- **Hafta 23:** Multi-Species Evolution
- **Hafta 24:** Hawk-Dove Oyunu'ndan AlphaStar'a
- **Hafta 25:** Mimicry ve Deception
- **Hafta 26:** Culture ve Convention Formation

### **Çeyrek 3: Asimetri ve Kompleks Dinamikler (Hafta 27-39)**

**Hafta 27-30: Bayesian Oyunlar**
- **Hafta 27:** Incomplete Information Games
- **Hafta 28:** Mechanism Design for MARL
- **Hafta 29:** Adversarial Inverse RL
- **Hafta 30:** Perfect Bayesian Equilibrium

**Hafta 31-34: Network Effects**
- **Hafta 31:** GNN Buluşuyor Network Games
- **Hafta 32:** Congestion Games ve Routing
- **Hafta 33:** Epidemic Models ve MARL
- **Hafta 34:** Influence Maximization Games

**Hafta 35-39: Stokastik Oyunlar**
- **Hafta 35:** Markov Games'in Derinlikleri
- **Hafta 36:** Differential Games
- **Hafta 37:** Stochastic Stability
- **Hafta 38:** Temporal Logic Specifications
- **Hafta 39:** Risk-Sensitive MARL

### **Çeyrek 4: Meta-Learning ve Frontier (Hafta 40-52)**

**Hafta 40-43: Meta-Learning**
- **Hafta 40:** Meta-Game Theory
- **Hafta 41:** Opponent Modeling'in Derinlikleri
- **Hafta 42:** Hypernetworks for Strategy Generation
- **Hafta 43:** Transfer Learning Across Games

**Hafta 44-47: Büyük Ölçek**
- **Hafta 44:** Mean Field Multi-Type Games
- **Hafta 45:** MCTS ve Extensive-Form Perfection
- **Hafta 46:** Sampling-Based Nash Equilibrium
- **Hafta 47:** Distributed MARL at Scale

**Hafta 48-52: Frontier & Kapanış**
- **Hafta 48:** Quantum Game Theory Meets MARL
- **Hafta 49:** Causal Inference in Multi-Agent Systems
- **Hafta 50:** Bounded Rationality ve Level-k Reasoning
- **Hafta 51:** MARL'in Büyük Açık Problemleri
- **Hafta 52:** Oyun Teorisi + MARL = AGI mi?
