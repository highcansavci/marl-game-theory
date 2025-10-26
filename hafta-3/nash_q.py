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