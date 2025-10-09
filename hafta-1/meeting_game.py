import torch
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

class MeetingGame:
    """İki arkadaş buluşmaya çalışıyor: Park (0) veya Kafe (1)"""
    def __init__(self):
        # Ödül matrisi: [ajan1_seçimi, ajan2_seçimi]
        self.rewards = {
            (0, 0): (10, 10),   # İkisi de Park -> Buluşma!
            (0, 1): (-5, -5),   # Biri Park, biri Kafe -> Buluşamama
            (1, 0): (-5, -5),   # Biri Kafe, biri Park -> Buluşamama  
            (1, 1): (10, 10),   # İkisi de Kafe -> Buluşma!
        }
    
    def step(self, actions):
        """Ajanlar karar veriyor, ödüller hesaplanıyor"""
        a1, a2 = actions
        r1, r2 = self.rewards[(a1, a2)]
        return r1, r2

class IndependentQLearner:
    """Her ajan bağımsız Q-learning yapıyor (diğerinden habersiz)"""
    def __init__(self, n_actions=2, lr=0.1, gamma=0.95, epsilon=0.1):
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = np.zeros(n_actions)  # Q-değerleri
    
    def choose_action(self):
        """Epsilon-greedy: Bazen keşfet, bazen en iyiyi seç"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)  # Keşif
        return np.argmax(self.Q)  # En iyi
    
    def learn(self, action, reward):
        """Q-learning güncellemesi"""
        # Bellman update: Q(a) <- Q(a) + lr * (r + gamma*max(Q) - Q(a))
        target = reward + self.gamma * np.max(self.Q)
        self.Q[action] += self.lr * (target - self.Q[action])

# Oyunu başlatalım!
game = MeetingGame()
agent1 = IndependentQLearner(epsilon=0.15)  # İlk ajan
agent2 = IndependentQLearner(epsilon=0.15)  # İkinci ajan

# Öğrenme süreci
n_episodes = 5000
history = {'rewards': [], 'actions1': [], 'actions2': []}

print("Öğrenme başlıyor... Her ajan bağımsız karar veriyor.\n")

for episode in range(n_episodes):
    # İki ajan da kendi Q-tablolarına bakarak karar veriyor
    a1 = agent1.choose_action()
    a2 = agent2.choose_action()
    
    # Oyun oynuyoruz ve sonuçları görüyoruz
    r1, r2 = game.step((a1, a2))
    
    # Her ajan kendi deneyiminden öğreniyor (diğerinden habersiz!)
    agent1.learn(a1, r1)
    agent2.learn(a2, r2)
    
    # İstatistik tutuyoruz
    history['rewards'].append((r1 + r2) / 2)
    history['actions1'].append(a1)
    history['actions2'].append(a2)

# Sonuçları görelim
print("=== Öğrenme Tamamlandı ===\n")
print(f"Ajan 1 Q-değerleri: Park={agent1.Q[0]:.2f}, Kafe={agent1.Q[1]:.2f}")
print(f"Ajan 2 Q-değerleri: Park={agent2.Q[0]:.2f}, Kafe={agent2.Q[1]:.2f}")

# Test: Son 1000 bölümde ne kadar koordine olabildiler?
last_actions = list(zip(history['actions1'][-1000:], history['actions2'][-1000:]))
coordination_rate = sum(1 for a1, a2 in last_actions if a1 == a2) / 1000
print(f"\nKoordinasyon Başarısı: %{coordination_rate*100:.1f}")
print(f"(Ne kadar sıklıkla aynı yeri seçtiler?)")

# Hangi dengeye yakınsadılar?
park_rate = sum(1 for a1, a2 in last_actions if a1 == 0 and a2 == 0) / 1000
cafe_rate = sum(1 for a1, a2 in last_actions if a1 == 1 and a2 == 1) / 1000
print(f"\nNash Dengesi Seçimi:")
print(f"  Park-Park: %{park_rate*100:.1f}")
print(f"  Kafe-Kafe: %{cafe_rate*100:.1f}")

# Görselleştirme
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Sol: Ortalama ödül zamanla
window = 100
smoothed = np.convolve(history['rewards'], np.ones(window)/window, mode='valid')
ax1.plot(smoothed, color='#2E86AB', linewidth=2)
ax1.axhline(y=10, color='green', linestyle='--', alpha=0.5, label='Optimal (Koordinasyon)')
ax1.axhline(y=-5, color='red', linestyle='--', alpha=0.5, label='En Kötü (Karışıklık)')
ax1.fill_between(range(len(smoothed)), -5, smoothed, alpha=0.2, color='#2E86AB')
ax1.set_xlabel('Bölüm (Episode)', fontsize=12)
ax1.set_ylabel('Ortalama Ödül', fontsize=12)
ax1.set_title('Zaman İçinde Koordinasyon Öğrenimi', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)

# Sağ: Son durumda Q-değerleri
locations = ['Park', 'Kafe']
x = np.arange(len(locations))
width = 0.35
ax2.bar(x - width/2, agent1.Q, width, label='Ajan 1', color='#A23B72', alpha=0.8)
ax2.bar(x + width/2, agent2.Q, width, label='Ajan 2', color='#F18F01', alpha=0.8)
ax2.set_ylabel('Q-Değeri (Beklenen Ödül)', fontsize=12)
ax2.set_title('Her Ajanın Tercih Değerleri', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(locations)
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('coordination_learning.png', dpi=150, bbox_inches='tight')
print("\n✓ Grafik kaydedildi: coordination_learning.png")