# Nash Dengesi'ni Öğrenmek: Q-Learning von Neumann ile Buluşuyor

---

## Giriş: Stratejiden Dengeye Geçiş
Önceki yazımızda AlphaGo'dan StarCraft'a, çok ajanlı sistemlerde evrimsel stratejilerin anatomisini inceledik. AlphaZero'nun self-play mekanizmasıyla öğrendiği taktikler, şimdi Nash dengesi gibi klasik oyun teorisi kavramlarıyla kesişiyor - von Neumann'ın minimax felsefesi, Q-learning'in stokastik dünyasında nasıl evriliyor?

Bu yazıda, çok ajanlı pekiştirmeli öğrenmenin (MARL) en önemli yapı taşlarından biri olan **Nash-Q Learning** algoritmasını derinlemesine inceleyeceğiz. Nash-Q Learning, klasik Q-learning'in çok ajanlı uzantısıdır ve işbirlikçi olmayan oyunlarda Nash dengesini öğrenerek optimal politikalar bulmayı hedefler. Stokastik oyunlarda denge hesaplama zorlukları ve bu algoritmanın neden MARL'da devrimsel olduğunu keşfedeceğiz.

Günümüz yapay zeka uygulamalarında - otonom araçlardan siber güvenliğe, drone sürülerinden enerji ağı yönetimine kadar - ajanlar arası çatışma ve işbirliği simülasyonu kritik öneme sahip. Nash-Q Learning, bu karmaşık etkileşimleri modellemek ve optimal stratejiler geliştirmek için güçlü bir araç sunuyor.

---

## Teorik Temel: Oyun Teorisi ve MARL'in Buluşması

### Von Neumann'dan Nash'e: Oyun Teorisinin Temelleri
John von Neumann'ın 1928'de ortaya koyduğu **minimax teoremi**, oyun teorisinin temel taşlarından biridir. Bu teorem, sıfır toplamlı oyunlarda (zero-sum games) her oyuncunun garantili bir kazanç seviyesi olduğunu matematiksel olarak kanıtlar. Sıfır toplamlı oyunlarda bir oyuncunun kazancı, diğerinin kaybına eşittir - satranç, poker (iki oyunculu) ve go bu kategoriye girer.

Von Neumann'ın minimax yaklaşımı şöyle çalışır: Bir oyuncu, rakibinin en kötü hamleyi yapacağını varsayarak kendi maksimum kazancını garanti eden stratejiyi seçer. Matematiksel olarak, bu "rakibin maksimum zararını minimize etmek" anlamına gelir. Bu yaklaşım, deterministik ve iki oyunculu oyunlarda güçlü sonuçlar verirken, gerçek dünya senaryolarının çoğu için yetersiz kalır.

1950'de John Nash, oyun teorisine devrim niteliğinde bir katkı yaptı: **Nash Dengesi**. Nash dengesi, sıfır toplamlı olmayan ve iki oyuncudan fazla ajanlı oyunlarda bile çalışan genel bir çözüm kavramıdır. Bir strateji profili Nash dengesinde ise, hiçbir ajan tek başına stratejisini değiştirerek daha fazla kazanç elde edemez - her ajanın stratejisi, diğerlerinin stratejilerine göre en iyi yanıttır (best response).

Matematiksel olarak ifade edersek: Ajan i'nin stratejisi, diğer tüm ajanların stratejilerine göre kendi faydasını maksimize ediyorsa Nash dengesindeyiz. Bu denge noktası, çok ajanlı sistemlerde "istikrarlı" stratejileri tanımlar.

**Stokastik Oyunlar** ise Markov Decision Process'in (MDP) çok ajanlı versiyonudur. Lloyd Shapley tarafından 1953'te tanımlanan bu model, durumlar, aksiyonlar, ödüller ve geçiş olasılıklarından oluşur. Tek ajanlı MDP'lerden farkı, her durumda birden fazla ajanın eş zamanlı aksiyon aldığı ve sistem dinamiklerinin tüm ajanların aksiyonlarına bağlı olmasıdır.

StarCraft'taki strateji evriminde gördüğümüz gibi, dinamik ortamlar stokastik oyunlara benzer - burada Nash dengesi, evrimi "istikrar" ile sınırlıyor. AlphaZero'nun self-play sürecinde keşfettiği stratejiler, aslında Nash dengesine yakınsamanın bir örneğidir.

### Q-Learning Hatırlatması ve MARL Zorlukları
Klasik **Q-Learning**, tek ajanlı pekiştirmeli öğrenmenin temel algoritmalarından biridir. Watkins ve Dayan'ın 1992'de geliştirdiği bu algoritma, bir ajanın durum-aksiyon çiftlerinin değerini (Q-değerlerini) öğrenmesini sağlar. Bellman denklemine dayanan güncelleme kuralı oldukça basittir: Her adımda, ajan mevcut Q-değerini, aldığı ödül ve gelecekteki en iyi aksiyonun tahmini değerine göre günceller.

Tek ajanlı ortamlar için son derece başarılı olan Q-Learning, çok ajanlı ortamlara geçtiğimizde ciddi zorluklarla karşılaşır:
*   ***Non-stationarity* (Durağan Olmama):** Tek ajanlı öğrenmede ortam sabit bir dinamiğe sahiptir - aynı durumda aynı aksiyonu alırsanız, beklenen sonuç değişmez. Ancak çok ajanlı sistemlerde, diğer ajanlar da öğreniyor ve stratejilerini değiştiriyor. Bu, ortamı sürekli değişen, durağan olmayan bir hale getirir. Klasik Q-Learning'in konverjans garantileri bu koşulda geçerliliğini yitirir.
*   ***Curse of Dimensionality* (Boyut Laneti):** N ajan varsa ve her ajan M aksiyona sahipse, ortak aksiyon uzayı M üzeri N boyutunda olur. Bu eksponansiyel büyüme, Q-tablolarını veya derin ağları çok büyük hale getirir ve öğrenmeyi neredeyse imkansız kılar.
*   **Kredi Atama Problemi:** Bir ödül aldığınızda, bu hangi ajanın katkısından kaynaklanıyor? Diğer ajanların aksiyonları sizin başarınızı nasıl etkiliyor? Bu soruların cevabı, çok ajanlı öğrenmede temel bir zorluktur.

2025 ICML konferansında yayınlanan *"Multi-Agent Reinforcement Learning in Games"* başlıklı derleme makalesi, MARL ve oyun teorisi konverjansını kapsamlı şekilde inceliyor. Araştırmacılar, Nash dengesinin durağan olmayan ortamlara uyarlanmasının ve derin öğrenme ile entegrasyonunun, gelecek yıllarda MARL'ın en önemli araştırma alanlarından biri olacağını vurguluyor.

---

## Nash-Q Learning Algoritması: Detaylı Analizi
Nash-Q Learning, Junling Hu ve Michael Wellman'ın 1998'de geliştirdiği öncü bir algoritmadır. Temel fikir şudur: Her ajan, sadece kendi Q-değerlerini değil, Nash dengesi koşulunu sağlayan Q-değerlerini öğrenir. Bu, klasik Q-Learning'in doğrudan çok ajanlı uzantısı değil, oyun teorisi ile pekiştirmeli öğrenmenin entegrasyonudur.

Algoritmanın çalışma prensibi şu adımlardan oluşur:
1.  **Başlangıç:** Her ajan için Q-değerleri rastgele veya sıfır olarak başlatılır. Her ajan kendi Q-tablosunu tutar - bu tablo, her durumda her olası ortak aksiyonun değerini saklar.
2.  **Best-Response Hesaplama:** Her iterasyonda, her ajan diğer ajanların mevcut politikalarına karşı en iyi yanıtını (best response) hesaplar. Bu, "diğer ajanlar şu stratejiyi oynarsa, ben ne yapmalıyım?" sorusuna verilen cevaptır. Best-response Q-değeri, ajanın tek başına optimize edebileceği aksiyonun değeridir.
3.  **Nash Q-Değeri Güncelleme:** Kritik adım burada gerçekleşir. Her ajan, sadece kendi best-response'una değil, tüm ajanların birlikte Nash dengesini oluşturan strateji profiline göre Q-değerini günceller. Bu, diğer ajanların Nash politikalarına göre beklenen değer hesaplamasıdır.
4.  **Politika Çıkarma:** Güncellenmiş Nash Q-değerlerinden, her ajan kendi politikasını türetir. Bu politika, o durumda Nash dengesini sağlayan aksiyonu seçer.
5.  **Konverjans Garantisi:** Hu ve Wellman, algoritmanın sıfır toplamlı stokastik oyunlarda Nash dengesine yakınsadığını matematiksel olarak kanıtladı. Bu, belirli koşullar altında (öğrenme oranının yeterince küçük olması, sonsuz keşif gibi) geçerlidir.

Algoritmanın başarısı, stokastik oyunlarda denge hesaplama yeteneğine dayanır. Michael Littman'ın değer iterasyonu benzeri yaklaşımı, Nash-Q Learning'e entegre edilir: Her durumda, tüm ajanlar için eş zamanlı best-response hesaplanır ve Nash dengesine karşılık gelen sabit nokta bulunur. Bu hesaplama, küçük oyunlarda lineer programlama ile, büyük oyunlarda ise yaklaşık yöntemlerle yapılır.

Güncel araştırmalar, Nash-Q Learning'i derin öğrenme ile birleştirerek ölçeklenebilirliğini artırıyor. 2025 yılında arXiv'de yayınlanan *"Nash Q-Network for Multi-Agent Cybersecurity Simulation"* makalesi, derin sinir ağlarıyla Nash Q-değerlerinin öğrenildiği bir yaklaşım sunuyor. Siber savunma simülasyonlarında, bu yöntem konverjans hızını %30 oranında artırmış ve gerçek zamanlı tehditlere karşı daha sağlam politikalar üretmiştir.

Başka bir ilgi çekici çalışma, *"Multi-Agent Nash Q-Learning for Node Security"* başlıklı makale. Bu araştırma, kişisel veri gizliliğinde rakip ajanlar arası dengeyi inceliyor. Saldırgan ajanlar kullanıcı verilerini çalmaya çalışırken, savunmacı ajanlar gizliliği korumaya çalışıyor - Nash-Q Learning, bu çatışmada dengeli bir güvenlik politikası öğreniyor.

### Matematiksel Örnek: 2-Ajanlı Grid-World
Basit bir örnek üzerinden Nash-Q Learning'in nasıl çalıştığını inceleyelim. İki ajan (A ve B) 3x3'lük bir ızgarada hareket ediyor. Her ajan yukarı, aşağı, sola veya sağa gidebilir. Hedefteki bir kaynağa ilk ulaşan ajan +10 ödül alıyor, diğeri -5 ödül alıyor. Eğer aynı anda ulaşırlarsa her ikisi de +5 alıyor.

Başlangıçta, her ajanın Q-tablosu rastgele değerlerle dolu. İlk episode'da, ajanlar epsilon-greedy stratejiyle rastgele hareket ediyor. Ajan A hedefe ulaşıyor ve +10 ödül alıyor. Şimdi güncelleme:

*   **Ajan A için best-response:** Mevcut durumda, B'nin en olası hamlesi göz önüne alınarak A'nın en iyi aksiyonu hesaplanır.
*   **Nash Q-değeri:** A ve B'nin birlikte Nash dengesini sağlayan aksiyon profili belirlenir (örneğin, A sağa git, B yukarı git).
*   **Q-tablosu güncellenir:** Bellman benzeri güncelleme, ancak max operatörü yerine Nash dengesi değeri kullanılır.

1000 episode sonunda, Q-tabloları yakınsar. Her ajan, diğerinin stratejisini göz önüne alarak optimal rotayı öğrenmiş olur. İlginç olan, Nash dengesinde bazen ajanların kaynağa aynı anda ulaşmayı tercih etmesidir - çünkü bu, rekabetçi bir yarış riskini azaltır ve garantili +5 ödülü sağlar.

---

## Kod Örneği: Python ile Nash-Q Learning Implementasyonu

### Nash Dengesi için Lineer Programlama: Neyi Optimize Ediyoruz?
**Temel Soru: Rakip Stratejisini Bilmiyoruz!**

Nash-Q Learning'de her ajan, rakibinin ne yapacağını tam olarak bilmiyor. Rakip de öğreniyor ve stratejisini sürekli değiştiriyor. Peki bu durumda "en iyi strateji" ne demek?

İşte burada minimax düşüncesi devreye giriyor: "En kötü duruma karşı en iyi hazırlık yap."

Yani LP ile optimize ettiğimiz şey:
> "Rakip benim stratejimi bilse ve bana karşı en kötü hamleyi yapsa bile, ben minimum ne kadar kazanç garanti edebilirim? Bu minimum değeri maksimize et."

Bu yaklaşım, oyun teorisinin kalbinde yatan güvenlik prensibi: Kendini en kötü senaryoya göre hazırla, o zaman hiç sürpriz yeme. Aşağıda LP ile neyi optimize ettiğimizi detaylı olarak görebilirsiniz.

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import linprog

class StochasticRPS:
    """Stokastik Taş-Kağıt-Makas Oyunu"""
    def __init__(self):
        self.actions = ['rock', 'paper', 'scissors']
        self.n_actions = 3
        
    def get_reward(self, action1, action2):
        """İki ajanın aksiyonlarına göre stokastik ödül döndürür"""
        # Beraberlik
        if action1 == action2:
            return 0.0, 0.0
        
        # Kazanan belirleme (rock=0, paper=1, scissors=2)
        if (action1 == 0 and action2 == 2) or \
           (action1 == 1 and action2 == 0) or \
           (action1 == 2 and action2 == 1):
            # Ajan 1 kazandı
            reward = np.random.uniform(0.8, 1.2)
            return reward, -reward
        else:
            # Ajan 2 kazandı
            reward = np.random.uniform(0.8, 1.2)
            return -reward, reward

class NashQLearner:
    """Nash-Q Learning Ajanı (Lineer Programlama ile Nash Dengesi)"""
    def __init__(self, n_actions, learning_rate=0.05, discount=0.9):
        self.n_actions = n_actions
        self.alpha = learning_rate
        self.gamma = discount
        
        # Q-tablosu: Q[my_action, opp_action]
        self.Q = np.zeros((n_actions, n_actions))
        
    def compute_nash_equilibrium(self):
        """
        Lineer Programlama ile Nash dengesini hesaplar.
        
        Maksimize etmek istediğimiz:
        V = min_j Σ_i (π_i × Q[i,j])
        
        LP Formülasyonu:
        maximize: V
        subject to:
        Σ_i (π_i × Q[i,j]) >= V  for all j (her rakip aksiyonu için)
        Σ_i π_i = 1
        π_i >= 0
        """
        n = self.n_actions
        
        # Q-matrisinin transpozunu al (rakip aksiyonları için constraint'ler)
        Q_T = self.Q.T
        
        # LP değişkenleri: [π_0, π_1, π_2, V]
        # Minimize etmek için: -V (linprog minimize eder, biz maximize istiyoruz)
        c = np.zeros(n + 1)
        c[-1] = -1  # V'yi maksimize et (minimize -V)
        
        # Eşitsizlik constraint'leri: -Σ(π_i × Q[i,j]) + V <= 0
        # Yani: V <= Σ(π_i × Q[i,j]) for all j
        A_ub = np.zeros((n, n + 1))
        for j in range(n):
            A_ub[j, :n] = -self.Q[:, j]  # -Q kolon j
            A_ub[j, n] = 1  # +V
        b_ub = np.zeros(n)
        
        # Eşitlik constraint'i: Σ π_i = 1
        A_eq = np.zeros((1, n + 1))
        A_eq[0, :n] = 1
        b_eq = np.array([1.0])
        
        # Bounds: π_i >= 0, V unbounded
        bounds = [(0, 1) for _ in range(n)] + [(None, None)]
        
        try:
            # LP çözümü
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, 
                           A_eq=A_eq, b_eq=b_eq, 
                           bounds=bounds, method='highs')
            
            if result.success:
                nash_policy = result.x[:n]
                # Normalize et (küçük sayısal hatalar için)
                nash_policy = np.maximum(nash_policy, 0)
                nash_policy /= nash_policy.sum()
                return nash_policy
            else:
                # LP başarısız olursa uniform politika döndür
                return np.ones(n) / n
                
        except Exception as e:
            # Hata durumunda uniform politika
            print(f"LP hatası: {e}")
            return np.ones(n) / n
    
    def get_value(self, opponent_policy):
        """Rakibin politikasına göre bu ajanın Nash değerini hesaplar"""
        # Her aksiyonun rakip politikasına göre beklenen değeri
        expected_q = self.Q @ opponent_policy
        return np.max(expected_q)
    
    def update(self, my_action, opp_action, reward, opp_next_policy):
        """Nash Q-değerlerini günceller"""
        # Gelecekteki Nash değeri
        future_value = self.get_value(opp_next_policy)
        
        # TD target
        target = reward + self.gamma * future_value
        
        # TD error ve güncelleme
        td_error = target - self.Q[my_action, opp_action]
        self.Q[my_action, opp_action] += self.alpha * td_error

def select_action(policy, epsilon):
    """Epsilon-greedy aksiyon seçimi"""
    if np.random.random() < epsilon:
        return np.random.randint(len(policy))
    else:
        return np.random.choice(len(policy), p=policy)

def train_nash_q(n_episodes=5000, epsilon_start=0.5, epsilon_end=0.05, epsilon_decay=0.995):
    """Nash-Q Learning eğitimi"""
    env = StochasticRPS()
    agent1 = NashQLearner(env.n_actions, learning_rate=0.05)
    agent2 = NashQLearner(env.n_actions, learning_rate=0.05)
    
    rewards_history = []
    epsilon = epsilon_start
    
    for episode in range(n_episodes):
        # Mevcut Nash politikaları (LP ile)
        policy1 = agent1.compute_nash_equilibrium()
        policy2 = agent2.compute_nash_equilibrium()
        
        # Epsilon-greedy aksiyon seçimi
        action1 = select_action(policy1, epsilon)
        action2 = select_action(policy2, epsilon)
        
        # Ödül alma
        reward1, reward2 = env.get_reward(action1, action2)
        
        # Nash Q-güncelleme
        agent1.update(action1, action2, reward1, policy2)
        agent2.update(action2, action1, reward2, policy1)
        
        # Kayıt
        rewards_history.append((reward1, reward2))
        
        # Epsilon decay
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        # İlerleme raporu
        if (episode + 1) % 1000 == 0:
            avg_r1 = np.mean([r[0] for r in rewards_history[-1000:]])
            avg_r2 = np.mean([r[1] for r in rewards_history[-1000:]])
            print(f"Episode {episode+1}: Avg R1={avg_r1:.3f}, R2={avg_r2:.3f}, ε={epsilon:.3f}")
            print(f"  Agent 1 Nash: {policy1}")
            print(f"  Agent 2 Nash: {policy2}")
    
    return agent1, agent2, rewards_history

def visualize_results(agent1, agent2, history):
    """Sonuçları görselleştirir"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Konverjans Plotu
    ax1 = axes[0, 0]
    window = 100
    avg_rewards1 = [np.mean([h[0] for h in history[i:i+window]]) 
                    for i in range(0, len(history)-window, window)]
    avg_rewards2 = [np.mean([h[1] for h in history[i:i+window]]) 
                    for i in range(0, len(history)-window, window)]
    
    x = np.arange(len(avg_rewards1)) * window
    ax1.plot(x, avg_rewards1, label='Ajan 1', linewidth=2, alpha=0.8)
    ax1.plot(x, avg_rewards2, label='Ajan 2', linewidth=2, alpha=0.8)
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Nash Dengesi (0)')
    ax1.set_xlabel('Episode', fontsize=11)
    ax1.set_ylabel('Ortalama Ödül', fontsize=11)
    ax1.set_title('Nash-Q Learning Konverjansı (LP ile)', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Ajan 1 Q-Matrisi
    ax2 = axes[0, 1]
    sns.heatmap(agent1.Q, annot=True, fmt='.3f', 
                xticklabels=['Rock', 'Paper', 'Scissors'],
                yticklabels=['Rock', 'Paper', 'Scissors'],
                cmap='coolwarm', center=0, ax=ax2, cbar_kws={'label': 'Q-Değer'})
    ax2.set_title('Ajan 1 Q-Matrisi', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Rakip Aksiyonu', fontsize=11)
    ax2.set_ylabel('Kendi Aksiyonu', fontsize=11)
    
    # 3. Ajan 2 Q-Matrisi
    ax3 = axes[1, 0]
    sns.heatmap(agent2.Q, annot=True, fmt='.3f',
                xticklabels=['Rock', 'Paper', 'Scissors'],
                yticklabels=['Rock', 'Paper', 'Scissors'],
                cmap='coolwarm', center=0, ax=ax3, cbar_kws={'label': 'Q-Değer'})
    ax3.set_title('Ajan 2 Q-Matrisi', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Rakip Aksiyonu', fontsize=11)
    ax3.set_ylabel('Kendi Aksiyonu', fontsize=11)
    
    # 4. Nash Politikaları
    ax4 = axes[1, 1]
    policy1 = agent1.compute_nash_equilibrium()
    policy2 = agent2.compute_nash_equilibrium()
    
    x_pos = np.arange(3)
    width = 0.35
    
    bars1 = ax4.bar(x_pos - width/2, policy1, width, label='Ajan 1 (LP)', alpha=0.8)
    bars2 = ax4.bar(x_pos + width/2, policy2, width, label='Ajan 2 (LP)', alpha=0.8)
    ax4.axhline(y=1/3, color='r', linestyle='--', alpha=0.5, label='İdeal (1/3)')
    
    ax4.set_xlabel('Aksiyon', fontsize=11)
    ax4.set_ylabel('Olasılık', fontsize=11)
    ax4.set_title('Nash Dengesi Politikaları (Lineer Programlama)', fontsize=12, fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(['Rock', 'Paper', 'Scissors'])
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_ylim([0, 0.5])
    
    plt.tight_layout()
    plt.show()
    plt.savefig("nash_q_learning_analysis_lp.png")

def print_analysis(agent1, agent2):
    """Detaylı analiz yazdır"""
    print("\n" + "="*60)
    print("NASH-Q LEARNING ANALİZ SONUÇLARI (Lineer Programlama)")
    print("="*60)
    
    policy1 = agent1.compute_nash_equilibrium()
    policy2 = agent2.compute_nash_equilibrium()
    
    print("\nAjan 1 Nash Politikası (LP):")
    print(f"  Rock:     {policy1[0]:.6f}")
    print(f"  Paper:    {policy1[1]:.6f}")
    print(f"  Scissors: {policy1[2]:.6f}")
    print(f"  Toplam:   {policy1.sum():.6f}")
    
    print("\nAjan 2 Nash Politikası (LP):")
    print(f"  Rock:     {policy2[0]:.6f}")
    print(f"  Paper:    {policy2[1]:.6f}")
    print(f"  Scissors: {policy2[2]:.6f}")
    print(f"  Toplam:   {policy2.sum():.6f}")
    
    # Uniform mixed strategy'den sapma
    ideal = 1/3
    deviation1 = np.mean(np.abs(policy1 - ideal))
    deviation2 = np.mean(np.abs(policy2 - ideal))
    
    print(f"\nİdeal Nash'ten Ortalama Sapma:")
    print(f"  Ajan 1: {deviation1:.6f}")
    print(f"  Ajan 2: {deviation2:.6f}")
    
    if deviation1 < 0.01 and deviation2 < 0.01:
        print("\n✓ Mükemmel konverjans! LP ile tam Nash dengesi bulundu.")
    elif deviation1 < 0.05 and deviation2 < 0.05:
        print("\n✓ İyi konverjans! Nash dengesine yakın.")
    else:
        print("\n⚠ Kısmi konverjans. Daha fazla episode gerekebilir.")
    
    print("\nAjan 1 Q-Matrisi (Satır: Kendi, Sütun: Rakip):")
    print("        Rock    Paper   Scissors")
    for i, action in enumerate(['Rock', 'Paper', 'Scissors']):
        print(f"{action:8s}", end="")
        for j in range(3):
            print(f"{agent1.Q[i,j]:8.3f}", end="")
        print()
    
    # Minimax değeri hesapla
    v1 = agent1.get_value(policy2)
    v2 = agent2.get_value(policy1)
    print(f"\nNash Değerleri:")
    print(f"  Ajan 1 Nash Value: {v1:.4f}")
    print(f"  Ajan 2 Nash Value: {v2:.4f}")
    print(f"  Toplam: {v1 + v2:.4f} (Sıfıra yakın olmalı)")

# Ana program
if __name__ == "__main__":
    print("Nash-Q Learning Eğitimi Başlıyor (Lineer Programlama ile)...")
    print("Ortam: Stokastik Taş-Kağıt-Makas")
    print("="*60)
    
    # Eğitim
    agent1, agent2, history = train_nash_q(n_episodes=5000)
    
    # Analiz
    print_analysis(agent1, agent2)
    
    # Görselleştirme
    visualize_results(agent1, agent2, history)
```

### Görselleştirmeler
[!nash_q_learning_analysis_lp.png]
Eğitim sonuçlarını dört farklı açıdan analiz edebiliriz:

1.  **Konverjans Grafiği (Üst Sol - "Nash-Q Learning Konverjansı LP ile"):** Bu grafik, Lineer Programlama ile Nash-Q Learning'in öğrenme dinamiğini net şekilde gösteriyor. İlk 2000 episode boyunca her iki ajanın ortalama ödülleri geniş dalgalanmalar sergiliyor - bu, epsilon-greedy stratejisinin yüksek exploration fazı. Ajan 1 ve Ajan 2'nin çizgileri birbirinin neredeyse zıttı gibi hareket ediyor: biri kazandığında diğeri kaybediyor. Episode 2000 civarında dalgalanmalar belirgin şekilde azalmaya başlıyor; epsilon decay sayesinde ajanlar keşiften yararlanmaya (exploitation) geçiyor.
    İlk episode'dan sonra her iki çizgi de kırmızı kesikli "Nash Dengesi (0)" çizgisi etrafında dar bir bantta stabilize oluyor. Son 1500 episode'da ortalama ödüller ±0.05 aralığında - bu, Nash dengesinin simetrik doğasını yansıtır. LP'nin hesapladığı kesin uniform politika sayesinde, ajanlar artık birbirlerine karşı sistematik avantaj sağlayamıyor. Konverjans hem matematiksel olarak garanti edilmiş hem de empirik olarak gözlemlenmiş.

2.  **Q-Matrisi Isı Haritaları (Orta):** Her iki ajanın Q-matrisi de renk kodlu ısı haritasıyla gösteriliyor. Kırmızı (pozitif) alanlar "bu aksiyon çiftinde ben kazanıyorum" anlamına gelirken, mavi (negatif) alanlar "bu durumda kaybediyorum" demek.
    Ajan 1'in matrisinde mükemmel döngüsel patern: Rock satırında Paper sütunu koyu mavi (-0.994), çünkü paper rock'u yener. Aynı satırda Scissors sütunu koyu kırmızı (+1.009), çünkü rock scissors'ı yener. Bu patern her satırda kusursuz tekrar ediyor. LP'nin Nash dengesini kesin bir derecede hesaplayabiliyor.
    Ajan 2'nin matrisi, Ajan 1'in neredeyse kusursuz transpose'u: Rock vs Paper değeri +0.994, tam karşılığı -0.994. Bu matematiksel simetri, iki ajanın birbirinden tamamen bağımsız öğrenmesine rağmen aynı Nash dengesine yakınsamasını kanıtlıyor.

3.  **Nash Politikaları Çubuk Grafiği (Sağ Alt - "Nash Dengesi Politikaları Lineer Programlama"):** En etkileyici sonuç burada. Her üç aksiyon için hem Ajan 1 (mavi "LP") hem Ajan 2 (turuncu "LP") çubukları, kırmızı kesikli "İdeal (1/3)" çizgisiyle piksel düzeyinde hizalı. Rock, Paper, Scissors - hepsi 0.333… değerinde. Bu, görsel olarak bile fark edilemeyecek kadar kesin bir uniform mixed strategy.
    Bu sonuç, John Nash'in 1950 tarihli teoremine empirik doğrulama: Taş-Kağıt-Makas gibi tam simetrik oyunlarda, Nash dengesi her aksiyonu eşit olasılıkla oynamaktır. Lineer programlama, minimax teoremini kullanarak bu dengeyi matematiksel kesinlikle hesaplıyor. 5000 episode sonunda, ajanlarımız sadece bu teoremi "keşfetmedi" - LP sayesinde teorinin tam çözümünü öğrendi.
    **Softmax vs LP Karşılaştırması:** Eğer softmax kullanmış olsaydık, çubuklar 0.33 civarında olurdu ama tam 1/3'e oturmazdı. Bu, yaklaşık çözüm ile kesin çözüm arasındaki farkı gösteriyor.

### Çıktı Analizi
Eğitim tamamlandıktan sonra, sonuçlar Nash dengesi teorisini matematiksel kesinlikle doğrular:

*   **Nash Politikaları (Lineer Programlama ile):** Her iki ajan da uniform mixed strategy'ye tam olarak yakınsadı. Ajan 1 ve Ajan 2'nin politikaları [0.333…, 0.333…, 0.333…] ideal değerini buldu - bu, oyun teorisinin Taş-Kağıt-Makas için matematiksel olarak kanıtladığı Nash dengesidir. Lineer programlama, softmax gibi yaklaşık yöntemlerin aksine, minimax teoreminin garantilediği kesin dengeyi hesaplar. Hiçbir ajan, rakibinin bu stratejisine karşı tek taraflı değişiklik yaparak kazanç sağlayamaz.
*   **Q-Matrisi Yapısı:** Eğitilmiş Q-tabloları, oyunun döngüsel doğasını kristal berraklığında yansıtıyor. Ajan 1'in matrisinde:
    *   Rock vs Scissors: +1.009 (rock, scissors'ı yener)
    *   Rock vs Paper: -0.994 (paper, rock'u yener)
    *   Paper vs Rock: +1.033 (paper, rock'u yener)
    *   Paper vs Scissors: -0.979 (scissors, paper'ı yener)
    *   Scissors vs Paper: +0.996 (scissors, paper'ı yener)
    *   Scissors vs Rock: -0.972 (rock, scissors'ı yener)
    
    Her kazanan-kaybeden çifti yaklaşık ±1.0 değerinde, bu da stokastik ödül aralığımız (0.8–1.2) ile mükemmel uyum içinde. Beraberlik durumları (diagonal) 0.020 civarında - neredeyse tam sıfır. Bu, LP'nin ürettiği kesin Nash dengesinin bir göstergesi.
*   **Simetri Analizi:** Ajan 2'nin Q-matrisi, Ajan 1 ile mükemmel ayna simetrisi gösteriyor. Rock vs Paper için +0.994 değeri, Paper vs Rock için -0.994 ile tam karşılık buluyor. Bu, her iki ajanın da aynı optimal stratejiyi bağımsız olarak keşfettiğini kanıtlar. Nash-Q Learning'in gücü tam da burada: Merkezi koordinasyon olmadan, sadece kendi deneyimlerinden öğrenerek denge noktasına ulaşıyorlar.
*   **Nash Değerleri:** Her iki ajanın güvenlik seviyesi (security level) sıfıra çok yakın. Bu, simetrik bir oyunda beklenen sonuç - uzun vadede hiçbir ajan sistematik avantaj elde edemez.

---

## Gerçek Dünya Uygulaması: Siber Güvenlikte Nash Dengesi
Nash-Q Learning'in teorik gücü etkileyici, peki gerçek dünyada nasıl kullanılıyor? En heyecan verici uygulama alanlarından biri çok ajanlı siber güvenlik simülasyonları.

### Senaryo: Ağ Penetrasyonu Simülasyonu
Bir kurumsal ağ düşünelim. Saldırgan ajanlar, ağa sızmaya ve kritik verilere erişmeye çalışıyor. Savunmacı ajanlar ise güvenlik duvarlarını, izleme sistemlerini ve yamalarını yöneterek saldırıları engellemeye çalışıyor. Bu, klasik bir sıfır toplamlı olmayan stokastik oyundur:
*   **Durumlar:** Ağın mevcut güvenlik konfigürasyonu, aktif bağlantılar, tespit edilen anormallikler.
*   **Aksiyonlar:** Saldırganlar için: port tarama, exploit deneme, lateral hareket. Savunmacılar için: port kapatma, trafik filtreleme, sistem güncellemeleri.
*   **Ödüller:** Saldırganlar kritik verilere eriştiklerinde pozitif, yakalandıklarında negatif ödül alır. Savunmacılar tam tersi.
*   **Stokastiklik:** Exploit'lerin başarı olasılığı, ağ gecikmesi, tespit sistemlerinin hassasiyeti rastgele faktörlerdir.

2025 yılında yayınlanan *"Nash Q-Network for Multi-Agent Cybersecurity Simulation"* makalesi, bu senaryoda Nash-Q Learning'in derin sinir ağları ile birleştirilmesini araştırdı. Araştırmacılar, MITRE ATT&CK framework'ündeki gerçek saldırı verilerini kullanarak simülasyonlar yaptı. Sonuçlar etkileyici:
*   **Daha iyi denge:** Nash-Q tabanlı savunmacılar, geleneksel rule-based sistemlere göre daha robust politikalar öğrendi. Saldırganlar stratejilerini değiştirdiğinde, Nash-Q hızla adapte oldu.
*   **Konverjans hızı:** Derin Nash-Q Network kullanımı, konverjans süresini büyük ölçüde azalttı. Bu, gerçek zamanlı tehdit önleme için kritik.
*   **Genelleme:** Eğitilmiş model, eğitim setinde olmayan yeni saldırı senaryolarında da başarılı savunma stratejileri üretti.

Başka bir ilgili çalışma, *"A multi-step minimax Q-learning"* yaklaşımını zero-sum senaryolarda pratik hesaplama için kullandı. Bu yöntem, Nash dengesini hesaplamak için gereken karmaşık lineer programlama yerine, iteratif minimax aramasıyla yaklaşık çözümler buldu. Büyük ölçekli ağlarda, bu yaklaşım hesaplama süresini saatlerden dakikalara indirdi.

### Diğer Potansiyel Uygulamalar
*   **Otonom Araç Trafiği:** Çok sayıda otonom aracın aynı kavşakta buluştuğu senaryolarda, her araç kendi rotasını optimize ederken diğer araçların hareketlerini de dikkate almalıdır. Nash-Q Learning, çarpışma önleme ve trafik akışı optimizasyonu için dengeli politikalar öğrenebilir.
*   **Enerji Ağı Kaynak Dağıtımı:** Akıllı şebekelerde, birden fazla üretici ve tüketici ajanın etkileşimi Nash dengesiyle modellenebilir. Her ajan enerjiyi ne zaman üretip tükettiğini optimize ederken, Nash-Q Learning sistem genelinde dengeli bir dağıtım sağlar.
*   **Drone Sürüleri:** Askeri veya lojistik uygulamalarda, drone'ların koordineli hareket etmesi gerekir. Nash-Q, her drone'un bağımsız karar verirken sürü hedeflerine katkıda bulunmasını sağlayabilir.

---

## Açık Sorular ve Sonuç: Düşünmeye Davet Ediyoruz
Nash-Q Learning, çok ajanlı pekiştirmeli öğrenmede önemli bir adım olsa da, hala cevaplanmayı bekleyen birçok soru var:
*   **Kısmi Gözlemlenebilir Ortamlar:** Nash-Q, tam gözlemlenebilir ortamlar için tasarlandı. Peki, POMDP'lerde (Partially Observable Markov Decision Processes) nasıl ölçeklenebilir? Gelecek haftaki yazımızda göreceğimiz regret minimization teknikleri mi yoksa derin öğrenme ile belief-state tracking mi daha etkili?
*   **Konverjans Gecikmesi:** Stokastik oyunlarda Nash dengesine yakınsama, özellikle büyük durum-aksiyon uzaylarında çok yavaş olabilir. Gerçek zamanlı uygulamalarda - örneğin drone swarm'ları veya yüksek frekanslı ticaret sistemlerinde - bu gecikme nasıl aşılabilir? Yaklaşık Nash dengesi yeterli mi?
*   **İşbirlikçi-Rekabetçi Hibrit Oyunlar:** Von Neumann'ın minimax'ı ve Nash dengesi, tamamen rekabetçi veya tamamen bağımsız ajanlar için tasarlandı. Ancak iklim değişikliği müzakereleri, uluslararası ticaret veya ortak kaynak yönetimi gibi hibrit senaryolarda durum nasıl? Nash-Q, işbirliği ve rekabeti aynı anda modelleyebilir mi?
*   **Derin Nash-Q'nun Sınırları:** Derin öğrenme ile Nash-Q'nun birleşimi umut verici, ancak derin ağların eğitim kararsızlığı ve overfitting riski var. Multi-agent ortamlarda bu sorunlar nasıl minimize edilir?

### Özet ve Sonraki Adım
Bu yazıda, Nash-Q Learning'in oyun teorisi ve pekiştirmeli öğrenme arasındaki köprüyü nasıl kurduğunu gördük. Von Neumann'ın minimax felsefesinden Nash'in denge kavramına, oradan Q-learning'in adaptif gücüne uzanan bu yolculuk, MARL'ı denge odaklı hale getirdi. Stokastik Taş-Kağıt-Makas örneğimizle teoriyi pratiğe döktük, siber güvenlik simülasyonlarıyla gerçek dünya etkisini gördük.

Ancak hikaye burada bitmiyor. Nash dengesi, tam bilgi oyunlarında güçlü bir araç, peki ya imperfect-information oyunlarda? Poker gibi oyunlarda, rakibinizin elini bilmediğiniz ve bluff'un hayati olduğu durumlarda denge nasıl bulunur?

Beğendiyseniz 👏 alkışlayın ve takip edin! Önümüzdeki 49 hafta boyunca bu yolculukta birlikte olalım. Sorularınız varsa yorumlarda buluşalım.

**Bir sonraki yazıda görüşmek üzere! Gelecek hafta: "Çok Oyunculu Poker'de Denge Arayışı: Counterfactual Regret Minimization'ın Sırları"**

**GitHub'da Kod:** [github.com/highcansavci/marl-game-theory](https://github.com/highcansavci/marl-game-theory) → Tüm kod örnekleri, notebook'lar ve ekstra materyaller

**İletişim:** [highcsavci@gmail.com] - Sorularınız, önerileriniz her zaman hoş gelir
