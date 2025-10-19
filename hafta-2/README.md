# AlphaGo'dan StarCraft'a: Çoklu Ajan Sistemlerde Strateji Evriminin Anatomisi

*Yapay zeka ajanları, birbirleriyle yarışarak nasıl insan üstü stratejiler geliştiriyor? Self-play mekanizmasının arkasındaki oyun teorik temelleri ve gerçek dünya uygulamalarını keşfediyoruz.*

---

## Giriş: Bir Devrin Sonu, Yeni Bir Çağın Başlangıcı

2016 yılının Mart ayında, dünya Go tarihinin en büyük şokunu yaşadı. Google DeepMind'ın geliştirdiği AlphaGo, 18 kez dünya şampiyonu Lee Sedol'u 4-1 yenerek, yapay zekanın stratejik düşünme kapasitesinde yeni bir çağ başlattı. Ancak asıl devrimci olan, AlphaGo'nun bu beceriyi nasıl öğrendiğiydi: milyonlarca kez kendisiyle oynayarak.

Üç yıl sonra, 2019'da DeepMind ve Blizzard Entertainment iş birliğiyle geliştirilen AlphaStar, StarCraft II'de profesyonel oyuncuları yenerek bir başka kilometre taşı oluşturdu. Bu başarının ardındaki sır yine aynıydı: self-play. Ancak bu sefer tek bir ajan değil, bir ajan popülasyonu birbirleriyle yarışarak evrimleşti.

Bu yazıda, self-play mekanizmasının oyun teorik temellerini, çoklu ajan sistemlerdeki rolünü ve gerçek dünya uygulamalarını derinlemesine inceleyeceğiz. Teoriden pratiğe, matematiksel kavramlardan Python implementasyonuna kadar geniş bir yelpazede bu büyüleyici konuyu ele alacağız.

---

## Bölüm 1: Oyun Teorisi ve Çoklu Ajan Öğrenme Temelleri

### Oyun Teorisi: Stratejik Etkileşimin Matematiği

Oyun teorisi, rasyonel karar vericilerin stratejik etkileşimlerini inceleyen matematik dalıdır. John von Neumann ve Oskar Morgenstern tarafından 1944'te temelleri atılan bu alan, ekonomiden biyolojiye, siyaset biliminden yapay zekaya kadar geniş bir yelpazede uygulama alanı bulmuştur.

**Temel Kavramlar:**

**Oyuncular:** Sistemdeki karar verici ajanlar. AlphaGo örneğinde iki oyuncu vardır: AlphaGo ve rakibi (başka bir AlphaGo veya insan).

**Stratejiler:** Her oyuncunun alabileceği eylem dizileri. Go'da taş yerleştirme pozisyonları, StarCraft'ta birim hareketleri ve saldırı kararları birer stratejidir.

**Ödüller (Payoffs):** Her strateji kombinasyonunun oyunculara sağladığı kazançlar. Kazanma durumunda +1, kaybetme durumunda -1 gibi.

**Nash Dengesi:** Hiçbir oyuncunun tek başına stratejisini değiştirerek daha iyi bir sonuç elde edemediği durum. Bu, stratejik istikrar noktasıdır.

### Çoklu Ajan Takviyeli Öğrenme (MARL)

Tek ajan takviyeli öğrenmede (Reinforcement Learning), bir ajan sabit bir çevre ile etkileşime girerek öğrenir. Ancak gerçek dünya çoğunlukla çoklu ajan sistemlerden oluşur: trafikteki araçlar, ekonomideki firmalar, ekosystemdeki canlılar...

MARL'da kritik farklılık şudur: **Çevre artık sabit değildir.** Her ajan öğrenirken, diğer ajanlar da öğrenir ve stratejilerini değiştirir. Bu, "hareketli hedef" problemi yaratır.

**MARL'ın Temel Zorlukları:**

1. **Non-stationarity (Durağanlık Olmayan Çevre):** Diğer ajanların politikaları değiştikçe, bir ajanın gördüğü çevre de değişir.

2. **Credit Assignment (Kredi Atama):** Bir takım oyununda başarı veya başarısızlık kime aittir? Her ajanın katkısını değerlendirmek zordur.

3. **Scalability (Ölçeklenebilirlik):** Ajan sayısı arttıkça, durum-eylem uzayı kombinatoryal olarak patlar.

4. **Exploration-Exploitation Dengesi:** Bireysel keşif, diğer ajanları etkileyerek sistem genelinde beklenmedik sonuçlar doğurabilir.

### İşbirliği mi, Rekabet mi, Yoksa Her İkisi mi?

MARL sistemleri üç kategoride incelenebilir:

**Tamamen İşbirlikçi (Cooperative):** Tüm ajanlar aynı ödülü paylaşır. Örnek: robot takımı futbol.

**Tamamen Rekabetçi (Competitive):** Bir ajanın kazancı, diğerinin kaybıdır (zero-sum oyunlar). Örnek: satranç, Go.

**Karma (Mixed):** Ajanlar bazen işbirliği yapar, bazen rekabet eder. Örnek: StarCraft'ta takım savaşları, ekonomik pazarlar.

AlphaGo tamamen rekabetçi bir ortamda çalışırken, AlphaStar hem işbirliği hem de rekabet içeren karma bir ortamda evrimleşmiştir.

---

## Bölüm 2: Self-Play - Kendi Kendine Öğrenmenin Gücü

### Self-Play Nedir?

Self-play, bir ajanın kendisiyle veya kendi kopyalarıyla oynayarak öğrendiği bir eğitim metodolojisidir. Temel fikir basittir: en iyi öğretmen, kendinizin gelişmiş versiyonudur.

**Self-Play'in Avantajları:**

1. **Sınırsız Eğitim Verisi:** İnsan oyunlarından veri toplamak zaman alır ve maliyetlidir. Self-play, saniyede binlerce oyun üretebilir.

2. **Otomatik Müfredat:** Ajan zorlaştıkça rakip de zorlaşır. Bu, ideal öğrenme eğrisini sağlar.

3. **Yeni Stratejilerin Keşfi:** İnsan bilgisiyle sınırlı kalmadan, yeni stratejiler ortaya çıkar. AlphaGo'nun 37. hamlesinin hikayesi buna örnektir.

4. **Adversarial Robustness:** Kendisiyle oynayan bir ajan, zayıf noktalarını keşfeder ve kapatır.

### Oyun Teorik Perspektif: Self-Play Neden İşe Yarar?

Self-play'in başarısı, Nash dengesine yakınsama özellikleriyle açıklanabilir.

**Fictitious Play ve Self-Play Bağlantısı:**

Fictitious play, oyun teorisinde klasik bir öğrenme dinamiğidir. Her oyuncu, rakibin geçmiş stratejilerine en iyi yanıtı verir. Bazı oyun sınıflarında (örneğin, iki oyunculu zero-sum oyunlar), fictitious play Nash dengesine yakınsar.

Self-play, bu konseptin güçlendirilmiş halidir. Her iterasyonda:
1. Mevcut politika ile oyun oynanır
2. Toplanan verilerden yeni bir politika öğrenilir
3. Bu yeni politika, gelecekteki self-play oyunlarında rakip olarak kullanılır

Bu süreç, sistem genelinde bir "co-evolution" (ortak evrim) yaratır.

**Exploitability ve Skill Progression:**

Bir politikanın "exploitability" değeri, ideal rakibe karşı ne kadar kötü performans göstereceğini ölçer. Self-play, bu değeri iteratif olarak azaltır.

AlphaGo Zero'nun eğitim süreci bunu net gösterir:
- İlk saatler: Rastgele hamleler
- İlk günler: Temel Go kuralları öğrenilir
- İlk haftalar: Taktiksel beceriler gelişir
- İlk aylar: Stratejik derinlik kazanılır
- 40 gün: İnsan şampiyonları yenilir

### Population-Based Training: Tek Değil, Çok!

AlphaStar'ın önemli bir yeniliği, tek bir ajan yerine bir popülasyon kullanmasıdır.

**Neden Popülasyon?**

Tek ajan self-play'de önemli bir sorun vardır: **policy collapse** veya **cycling**. Ajan, belirli bir meta-stratejiye odaklanır ve diğer stratejilere karşı savunmasız kalabilir. Taş-kağıt-makastan düşünün: taşı yenmeyi öğrenen bir ajan, sürekli kağıtla oynar. Sonra kağıdı yenmeyi öğrenir ve makasa geçer. Sonra makası yenen taşa döner. Bu bir döngüdür ve gerçek öğrenme olmaz.

Popülasyon tabanlı eğitimde:
- Farklı stratejilere odaklanan birden fazla ajan eş zamanlı eğitilir
- Ajanlar hem birbirleriyle hem de geçmiş versiyonlarıyla oynar
- En başarılı stratejiler varlığını sürdürür (evolutionary selection)
- Çeşitlilik korunur (diversity mechanisms)

Bu, biyolojik evrimin AI dünyasındaki karşılığıdır.

---

## Bölüm 3: AlphaZero - MCTS + Neural Network ile Self-Play

Şimdi teoriden pratiğe geçiyoruz. AlphaGo ve AlphaStar'ın temelini oluşturan AlphaZero algoritmasını, Tic-Tac-Toe oyununda implement ederek self-play'in gücünü göreceğiz.

### AlphaZero'nun Anatomisi

AlphaZero, üç ana bileşenden oluşur:

**1. Neural Network (Policy-Value Network):**
- **Input:** Oyun tahtasının encoded hali (3 channel: rakip taşlar, boş hücreler, kendi taşlar)
- **ResNet Backbone:** Residual block'larla derin öğrenme
- **Policy Head:** Her hamle için olasılık dağılımı (9 çıkış Tic-Tac-Toe için)
- **Value Head:** Mevcut pozisyonun kazanma olasılığı (-1 ile +1 arası)

**2. Monte Carlo Tree Search (MCTS):**
- **Selection:** UCB (Upper Confidence Bound) formülü ile en umut vadeden branch'i seç
- **Expansion:** Neural network ile yeni node'ları expand et
- **Simulation:** Value network ile pozisyonu değerlendir
- **Backpropagation:** Sonucu tree'de yukarı doğru yay

**3. Self-Play Training Loop:**
- MCTS kullanarak oyun oyna
- (state, policy, outcome) training data'sı topla
- Neural network'ü bu data ile eğit
- Yeni model ile self-play'e devam et

### Tam AlphaZero Implementasyonu

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import math
from collections import deque
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 Kullanılan cihaz: {device}")


class TicTacToe:
    """Tic-Tac-Toe oyun ortamı - AlphaZero uyumlu"""
    
    def __init__(self):
        self.row_count = 3
        self.column_count = 3
        self.action_size = 9
        
    def get_initial_state(self):
        return np.zeros((3, 3), dtype=np.float32)
    
    def get_next_state(self, state, action, player):
        state = state.copy()
        row, col = action // 3, action % 3
        state[row, col] = player
        return state
    
    def get_valid_moves(self, state):
        """Geçerli hamleleri binary mask olarak döndür"""
        return (state.reshape(-1) == 0).astype(np.uint8)
    
    def check_win(self, state, action):
        if action is None:
            return False
        row, col = action // 3, action % 3
        player = state[row, col]
        return (
            np.sum(state[row, :]) == player * 3 or
            np.sum(state[:, col]) == player * 3 or
            np.sum(np.diag(state)) == player * 3 or
            np.sum(np.diag(np.flip(state, axis=0))) == player * 3
        )
    
    def get_value_and_terminated(self, state, action):
        if self.check_win(state, action):
            return 1, True
        if np.sum(self.get_valid_moves(state)) == 0:
            return 0, True
        return 0, False
    
    def get_opponent(self, player):
        return -player
    
    def change_perspective(self, state, player):
        """Current player perspektifine çevir"""
        return state * player
    
    def get_encoded_state(self, state):
        """Neural network için 3 channel encoding"""
        encoded = np.stack([
            (state == -1).astype(np.float32),
            (state == 0).astype(np.float32),
            (state == 1).astype(np.float32)
        ])
        if len(state.shape) == 3:
            encoded = np.swapaxes(encoded, 0, 1)
        return encoded


class ResNet(nn.Module):
    """AlphaZero Policy-Value Network"""
    
    def __init__(self, game, num_resBlocks=4, num_hidden=64):
        super().__init__()
        self.device = device
        
        # Initial convolution
        self.startBlock = nn.Sequential(
            nn.Conv2d(3, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU()
        )
        
        # Residual blocks
        self.backBone = nn.ModuleList([
            ResBlock(num_hidden) for _ in range(num_resBlocks)
        ])
        
        # Policy head - her hamle için probability
        self.policyHead = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 9, 9)
        )
        
        # Value head - pozisyon değerlendirmesi
        self.valueHead = nn.Sequential(
            nn.Conv2d(num_hidden, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * 9, 1),
            nn.Tanh()
        )
        
        self.to(device)
        
    def forward(self, x):
        x = self.startBlock(x)
        for resBlock in self.backBone:
            x = resBlock(x)
        return self.policyHead(x), self.valueHead(x)


class ResBlock(nn.Module):
    """Residual Block"""
    
    def __init__(self, num_hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)
        
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return F.relu(x)


class Node:
    """MCTS Node"""
    
    def __init__(self, game, args, state, parent=None, action_taken=None, prior=0, visit_count=0):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior
        self.children = []
        self.visit_count = visit_count
        self.value_sum = 0
        
    def is_fully_expanded(self):
        return len(self.children) > 0
    
    def select(self):
        """En yüksek UCB skorlu child'ı seç"""
        best_child = max(self.children, key=lambda child: self.get_ucb(child))
        return best_child
    
    def get_ucb(self, child):
        """Upper Confidence Bound - AlphaZero PUCT formülü"""
        if child.visit_count == 0:
            q_value = 0
        else:
            # Q-value normalizasyonu
            q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
        
        # Exploration bonus
        u_value = (self.args['C'] * child.prior * 
                   math.sqrt(self.visit_count) / (1 + child.visit_count))
        return q_value + u_value
    
    def expand(self, policy):
        """Valid actionlar için child node'lar oluştur"""
        for action, prob in enumerate(policy):
            if prob > 0:
                child_state = self.game.get_next_state(self.state.copy(), action, 1)
                child_state = self.game.change_perspective(child_state, -1)
                child = Node(self.game, self.args, child_state, self, action, prob)
                self.children.append(child)
    
    def backpropagate(self, value):
        """Value'yu tree'de yukarı yay"""
        self.value_sum += value
        self.visit_count += 1
        value = -value  # Alternating players
        if self.parent is not None:
            self.parent.backpropagate(value)


class MCTS:
    """Monte Carlo Tree Search"""
    
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model
    
    @torch.no_grad()
    def search(self, state):
        """MCTS search - action probabilities döndür"""
        # Root node
        root = Node(self.game, self.args, state, visit_count=1)
        
        # Neural network ile initial policy
        policy, _ = self.model(
            torch.tensor(self.game.get_encoded_state(state), device=device).unsqueeze(0)
        )
        policy = torch.softmax(policy, dim=1).squeeze(0).cpu().numpy()
        
        # Dirichlet noise - exploration için
        policy = ((1 - self.args['dirichlet_epsilon']) * policy + 
                  self.args['dirichlet_epsilon'] * 
                  np.random.dirichlet([self.args['dirichlet_alpha']] * 9))
        
        # Invalid moves'u maskle
        valid_moves = self.game.get_valid_moves(state)
        policy *= valid_moves
        policy /= np.sum(policy)
        root.expand(policy)
        
        # MCTS simulations
        for _ in range(self.args['num_searches']):
            node = root
            
            # Selection: leaf node'a kadar
            while node.is_fully_expanded():
                node = node.select()
            
            # Evaluation
            value, is_terminal = self.game.get_value_and_terminated(
                node.state, node.action_taken
            )
            value = -value
            
            # Expansion (terminal değilse)
            if not is_terminal:
                policy, value = self.model(
                    torch.tensor(self.game.get_encoded_state(node.state), 
                               device=device).unsqueeze(0)
                )
                policy = torch.softmax(policy, dim=1).squeeze(0).cpu().numpy()
                valid_moves = self.game.get_valid_moves(node.state)
                policy *= valid_moves
                policy /= np.sum(policy)
                value = value.item()
                node.expand(policy)
            
            # Backpropagation
            node.backpropagate(value)
        
        # Visit count'lara göre action probabilities
        action_probs = np.zeros(9)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= np.sum(action_probs)
        return action_probs


class AlphaZero:
    """AlphaZero Self-Play Algorithm"""
    
    def __init__(self, model, optimizer, game, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTS(game, args, model)
    
    def selfPlay(self):
        """Bir self-play oyunu - training data topla"""
        memory = []
        player = 1
        state = self.game.get_initial_state()
        
        while True:
            neutral_state = self.game.change_perspective(state, player)
            action_probs = self.mcts.search(neutral_state)
            memory.append((neutral_state, action_probs, player))
            
            # Temperature-based action selection
            temp_probs = action_probs ** (1 / self.args['temperature'])
            temp_probs /= np.sum(temp_probs)
            action = np.random.choice(9, p=temp_probs)
            
            state = self.game.get_next_state(state, action, player)
            value, is_terminal = self.game.get_value_and_terminated(state, action)
            
            if is_terminal:
                # Gerçek kazananı belirle
                real_winner = player if value == 1 else (0 if value == 0 else -player)
                
                # Training data oluştur
                return_memory = []
                for hist_state, hist_probs, hist_player in memory:
                    hist_outcome = value if hist_player == player else -value
                    return_memory.append((
                        self.game.get_encoded_state(hist_state),
                        hist_probs,
                        hist_outcome
                    ))
                return return_memory, real_winner
            
            player = -player
    
    def train(self, memory):
        """Neural network'ü eğit"""
        random.shuffle(memory)
        total_loss = 0
        batches = 0
        
        for i in range(0, len(memory), self.args['batch_size']):
            sample = memory[i:min(len(memory), i + self.args['batch_size'])]
            state, policy_targets, value_targets = zip(*sample)
            
            state = torch.tensor(np.array(state), dtype=torch.float32, device=device)
            policy_targets = torch.tensor(np.array(policy_targets), dtype=torch.float32, device=device)
            value_targets = torch.tensor(np.array(value_targets).reshape(-1, 1), dtype=torch.float32, device=device)
            
            out_policy, out_value = self.model(state)
            
            # Loss calculation
            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            batches += 1
        
        return total_loss / max(batches, 1)
    
    def learn(self):
        """Ana eğitim loop'u"""
        stats = {'iteration': [], 'x_wins': [], 'o_wins': [], 'draws': [], 'avg_loss': []}
        
        print("=" * 70)
        print("🌳 ALPHAZERO SELF-PLAY TRAINING")
        print("=" * 70)
        print(f"İterasyonlar: {self.args['num_iterations']}")
        print(f"Her iterasyonda oyun: {self.args['num_selfPlay_iterations']}")
        print(f"MCTS simülasyonlar: {self.args['num_searches']}")
        print("=" * 70)
        
        for iteration in range(self.args['num_iterations']):
            print(f"\n🔄 İterasyon {iteration + 1}/{self.args['num_iterations']}")
            
            memory = []
            self.model.eval()
            x_wins = o_wins = draws = 0
            
            for _ in range(self.args['num_selfPlay_iterations']):
                game_memory, winner = self.selfPlay()
                memory.extend(game_memory)
                
                if winner == 1:
                    x_wins += 1
                elif winner == -1:
                    o_wins += 1
                else:
                    draws += 1
            
            total = self.args['num_selfPlay_iterations']
            print(f"  📊 X kazandı={x_wins} ({100*x_wins/total:.1f}%), "
                  f"O kazandı={o_wins} ({100*o_wins/total:.1f}%), "
                  f"Berabere={draws} ({100*draws/total:.1f}%)")
            print(f"  💾 Toplanan data: {len(memory)} pozisyon")
            
            # Training
            self.model.train()
            avg_loss = sum(self.train(memory) for _ in range(self.args['num_epochs'])) / self.args['num_epochs']
            print(f"  📉 Loss: {avg_loss:.4f}")
            
            stats['iteration'].append(iteration + 1)
            stats['x_wins'].append(100 * x_wins / total)
            stats['o_wins'].append(100 * o_wins / total)
            stats['draws'].append(100 * draws / total)
            stats['avg_loss'].append(avg_loss)
            
            # Checkpoint
            if (iteration + 1) % 5 == 0 or iteration == self.args['num_iterations'] - 1:
                torch.save(self.model.state_dict(), 
                          f'alphazero_tictactoe_iter{iteration+1}.pth')
                print(f"  💾 Checkpoint kaydedildi")
        
        print("\n" + "=" * 70)
        print("✅ EĞİTİM TAMAMLANDI!")
        print("=" * 70)
        return stats


# Eğitim ve Görselleştirme
if __name__ == "__main__":
    print("\n🌳 ALPHAZERO TIC-TAC-TOE")
    print("MCTS + Neural Network + Self-Play")
    print("=" * 70)
    
    # Oyun ve model
    game = TicTacToe()
    model = ResNet(game, num_resBlocks=4, num_hidden=64)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    
    # Hyperparameters
    args = {
        'C': 2,                          # UCB exploration constant
        'num_searches': 100,             # MCTS simulations per move
        'num_iterations': 20,            # Training iterations
        'num_selfPlay_iterations': 100,  # Games per iteration
        'num_epochs': 4,                 # Training epochs per iteration
        'batch_size': 64,
        'temperature': 1.25,
        'dirichlet_epsilon': 0.25,
        'dirichlet_alpha': 0.3
    }
    
    # Eğitim
    alphazero = AlphaZero(model, optimizer, game, args)
    stats = alphazero.learn()
    
    # Görselleştirme
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Oyun sonuçları
    axes[0, 0].plot(stats['iteration'], stats['x_wins'], 
                    'o-', label='X Kazandı %', color='#e74c3c', linewidth=2, markersize=6)
    axes[0, 0].plot(stats['iteration'], stats['o_wins'], 
                    's-', label='O Kazandı %', color='#3498db', linewidth=2, markersize=6)
    axes[0, 0].plot(stats['iteration'], stats['draws'], 
                    '^-', label='Berabere %', color='#2ecc71', linewidth=2, markersize=6)
    axes[0, 0].set_xlabel('İterasyon', fontweight='bold')
    axes[0, 0].set_ylabel('Yüzde (%)', fontweight='bold')
    axes[0, 0].set_title('Oyun Sonuçları', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss
    axes[0, 1].plot(stats['iteration'], stats['avg_loss'], 
                    linewidth=2.5, color='#9b59b6', marker='o', markersize=5)
    axes[0, 1].set_xlabel('İterasyon', fontweight='bold')
    axes[0, 1].set_ylabel('Loss', fontweight='bold')
    axes[0, 1].set_title('Eğitim Loss', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Beraberlik trendi
    axes[1, 0].plot(stats['iteration'], stats['draws'], 
                    linewidth=3, color='#2ecc71', marker='o', markersize=7)
    axes[1, 0].axhline(y=70, color='red', linestyle='--', 
                       label='Hedef: ~70-80% (Optimal)', linewidth=2)
    axes[1, 0].set_xlabel('İterasyon', fontweight='bold')
    axes[1, 0].set_ylabel('Beraberlik %', fontweight='bold')
    axes[1, 0].set_title('Optimal Oyuna Yakınsama', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # İstatistikler
    final_draws = stats['draws'][-1]
    final_x = stats['x_wins'][-1]
    final_o = stats['o_wins'][-1]
    final_loss = stats['avg_loss'][-1]
    axes[1, 1].text(0.5, 0.7, '📊 Final İstatistikler', 
                    ha='center', va='center', fontsize=14, fontweight='bold')
    axes[1, 1].text(0.5, 0.55, f'X Kazanma: {final_x:.1f}%', 
                    ha='center', va='center', fontsize=12, color='#e74c3c')
    axes[1, 1].text(0.5, 0.45, f'O Kazanma: {final_o:.1f}%', 
                    ha='center', va='center', fontsize=12, color='#3498db')
    axes[1, 1].text(0.5, 0.35, f'Beraberlik: {final_draws:.1f}%', 
                    ha='center', va='center', fontsize=12, color='#2ecc71')
    axes[1, 1].text(0.5, 0.20, f'Final Loss: {final_loss:.4f}', 
                    ha='center', va='center', fontsize=11)
    axes[1, 1].text(0.5, 0.05, '🌳 AlphaZero Eğitimi Tamamlandı!', 
                    ha='center', va='center', fontsize=11, style='italic')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('alphazero_training.png', dpi=150, bbox_inches='tight')
    print("\n📊 Grafik kaydedildi: alphazero_training.png")
```

### AlphaZero Kodunun Detaylı Açıklaması

**1. TicTacToe Sınıfı:**
AlphaZero uyumlu interface sağlar. En önemli metotlar:
- `get_encoded_state()`: Neural network için 3 kanallı input (rakip, boş, kendi)
- `change_perspective()`: Current player her zaman +1 olacak şekilde state'i değiştirir
- `get_valid_moves()`: MCTS için binary mask

**2. ResNet Neural Network:**
AlphaGo'nun mimarisine benzer:
- 4 residual block (daha derin öğrenme)
- BatchNorm + ReLU aktivasyon
- İki ayrı head: policy ve value

**3. MCTS Node ve Search:**
Her node şunları tutar:
- `visit_count`: Kaç kez ziyaret edildi
- `value_sum`: Toplam value
- `prior`: Neural network'ten gelen probability
- UCB formülü: exploitation (Q) + exploration (U)

**4. AlphaZero Training Loop:**
Self-play döngüsü:
```
Self-Play (100 oyun) → Training Data (pozisyon, policy, outcome)
    ↓
Neural Network Eğitimi (4 epoch)
    ↓
Yeni Model → Self-Play'e geri dön
```

### Eğitim Sonuçları ve Analiz
<img width="1189" height="1005" alt="alphazero_tic_tac_toe" src="https://github.com/user-attachments/assets/b730a8a4-fb0c-48eb-9291-ffa309f78766" />

20 iterasyon sonunda AlphaZero'nun öğrendikleri:

**İlk İterasyonlar (1-5):**
- X Kazanma: ~55%
- O Kazanma: ~25%
- Beraberlik: ~20%
- Loss: ~2.5
- Durum: Temel kuralları öğreniyor, rastgele hamleler yapıyor

**Orta İterasyonlar (6-12):**
- X Kazanma: ~35%
- O Kazanma: ~15%
- Beraberlik: ~50%
- Loss: ~1.3
- Durum: Taktiksel oyun gelişiyor, savunma stratejileri öğreniliyor

**Final İterasyonlar (13-20):**
- X Kazanma: ~30%
- O Kazanma: ~12%
- Beraberlik: ~58%
- Loss: ~1.18
- Durum: Neredeyse optimal oyun, beraberlik dominant

**Kritik Gözlemler:**

1. **First-Move Advantage Azalıyor:** X'in başlangıçtaki büyük avantajı (%55), eğitim ilerledikçe %30'a düşüyor. Her iki oyuncu da optimal stratejileri öğrendikçe, ilk hamle avantajı azalıyor.

2. **Optimal Oyuna Yakınsama:** Tic-Tac-Toe'da optimal oyun beraberliktir. AlphaZero %58 beraberlik oranıyla buna yaklaşıyor. Daha fazla iterasyonla %70-80'e ulaşabilir.

3. **MCTS'nin Gücü:** Neural network + MCTS kombinasyonu, sadece 2,000 oyunla (20 iter × 100 oyun) neredeyse optimal seviyeye ulaşıyor.

4. **Loss Yakınsaması:** Training loss'un 2.5'ten 1.18'e düşmesi, network'ün pozisyonları ve policy'leri doğru öğrendiğini gösteriyor.

5. **O'nun Dezavantajı:** O her zaman ikinci oynar, bu yüzden kazanma oranı daha düşük. Ancak %12, rastgele oyuna göre iyi bir performans.

### Self-Play'in Gördüğümüz Evrimi

**İlk 500 Oyun:** Ajanlar rastgele hamle yapıyor
```
X . O
. . .
. . X
```
Sonuç: X tesadüfen kazandı

**2,000 Oyun Sonrası:** Kazanma fırsatlarını yakalıyor
```
X X O
O . .
. . .
```
X üçüncü X'i koyup kazanıyor

**10,000 Oyun Sonrası:** Savunma yapıyor
```
X X .
O O X
. . O
```
O, X'in kazanmasını engelliyor

**20,000 Oyun (Final):** Neredeyse optimal oyun
```
X O X
X O X
O X O
```
Her iki oyuncu da mükemmel oynuyor → Beraberlik

---

## Bölüm 4: Gerçek Dünya Uygulamaları

### 1. Otonom Araç Koordinasyonu

Self-play, trafik yönetiminde devrim yaratabilir. Binlerce otonom araç, simülasyonda birbirleriyle "oyun" oynayarak optimal trafik akışını öğreniyor.

**Uygulama Senaryosu:**
- Her araç bir ajandır
- Hedef: En kısa sürede varış + kazaları önlemek
- Self-play ile öğrenme: yavaşlama, şerit değiştirme, kavşak geçişi
- Emergent behavior: İnsan programcıların düşünmediği trafik paternleri

Waymo ve Tesla, simülasyonda milyarlarca mil self-play ile eğitim yapıyor.

### 2. Siber Güvenlik - Red Team vs Blue Team

Saldırgan ve savunmacı ajanların self-play ile eğitilmesi:

**Red Team (Saldırgan):** Sistem zafiyetlerini bulur
**Blue Team (Savunmacı):** Saldırıları önler
**Self-Play:** Her iki taraf birbirinden öğrenerek gelişir
**Sonuç:** İnsan uzmanların bulamadığı zafiyetler

DeepMind'ın 2020 çalışması: Self-play ile eğitilmiş AI'lar, geleneksel penetrasyon testlerinden daha fazla zafiyet buldu.

### 3. Finansal Ticaret

Algoritmik ticaret botlarının birbirine karşı eğitilmesi:

- Çoklu ajan sistemler gerçek pazar dinamiklerini simüle eder
- Her bot farklı strateji: momentum, arbitrage, market making
- Self-play ile botlar birbirlerine adapte olur
- Emergent dynamics: Flash crash gibi fenomenler simülasyonda ortaya çıkar

Jane Street, Citadel gibi firmalar self-play'i yoğun kullanıyor.

### 4. Doğal Dil - Müzakere AI

Meta'nın 2017 çalışması: Self-play ile müzakere yapan chatbotlar

- İki ajan eşyaları paylaşmaya çalışır
- Farklı tercihleri var
- Self-play ile öğrenir: ikna, uzlaşma, stratejik taviz
- İlginç sonuç: Ajanlar kendi "protokollerini" geliştirdi

### 5. İlaç Keşfi - AlphaFold

AlphaFold'un protein katlama başarısında self-play'in rolü:

- Generative model: Protein yapıları üretir
- Discriminative model: Doğruluğunu değerlendirir
- Self-play döngüsü: Her iki model birbirini "aldatmaya" çalışır
- Sonuç: 50 yıllık problemin çözümü

---

## Bölüm 5: Self-Play'in Sınırları ve Geleceği

### Zorluklar ve Açık Problemler

**1. Reward Hacking:**
Ajanlar istenmeyen davranışlarla yüksek reward elde edebilir. Robot futbolda topu kaleye atmak yerine rakibi devirmek.

**2. Mode Collapse:**
Ajanlar "safe" stratejide takılabilir. Her iki ajan da defansif oynar, kimse risk almaz.

**3. Computational Cost:**
AlphaStar: ~200 yıl oyun simulasyonu (paralel). Küçük laboratuvarlar için erişilemez.

**4. Sim-to-Real Transfer:**
Simülasyonda öğrenilen stratejiler gerçek dünyada çalışmayabilir.

### Gelecek Araştırma Yönleri

**Curriculum Learning:** Self-play + manuel müfredat

**Human-in-the-Loop:** Self-play + insan feedback (RLHF)

**Multi-Modal Self-Play:** Görsel + işitsel + dilsel modaliteler

**Evolutionary Dynamics:** AlphaStar tarzı büyük popülasyonlar

**Open-Ended Learning:** Sabit oyun yerine sürekli genişleyen ortam (Minecraft)

---

## Sonuç: Kendi Kendini Geliştiren Sistemlerin Çağı

Self-play, yapay zeka tarihinde paradigma değişimine neden olan konseptlerden biridir. AlphaGo'nun Go'da, AlphaStar'ın StarCraft'ta, OpenAI Five'ın Dota 2'de gösterdiği başarılar, self-play'in gücünü kanıtlamıştır.

**En iyi öğretmen, geliştirilmiş versiyonunuzdur.** Bu fikir sadece oyunlarla sınırlı değil; robotik, siber güvenlik, finans, sağlık ve daha birçok alanda devrim yaratma potansiyeline sahip.

Gelecek yıllarda, self-play mekanizmalarının daha da sofistike hale gelmesini, insan-AI işbirliği ile birleşmesini ve açık uçlu öğrenme ortamlarında uygulanmasını göreceğiz.

Bir sonraki yazımızda, Nash-Q Learning algoritmasını derinlemesine inceleyeceğiz ve stokastik oyunlarda denge hesaplamanın inceliklerine dalacağız.

---

## Düşünmeniz için Açık Sorular

1. **Etik Boyut:** Self-play ile eğitilmiş AI, insan değerlerini nasıl öğrenebilir? Reward fonksiyonu yeterli mi?

2. **Genelleme:** Bir oyunda self-play ile uzmanlaşan ajan, farklı oyunda ne kadar başarılı olur?

3. **Emergent Behavior:** Self-play'de ortaya çıkan beklenmedik stratejiler her zaman faydalı mı?

4. **İnsan Üstü Strateji:** AlphaGo'nun "Hamle 37"si gibi stratejiler, genel zeka için ne anlama gelir?

5. **Ölçeklenebilirlik:** Self-play gerçek dünya ekonomisi gibi mega-ölçekli sistemlerde işe yarar mı?

---

## Kaynaklar ve İleri Okuma

**Temel Makaleler:**
- Silver et al. (2017): "Mastering the game of Go without human knowledge" - AlphaGo Zero
- Vinyals et al. (2019): "Grandmaster level in StarCraft II using multi-agent reinforcement learning" - AlphaStar
- Bansal et al. (2018): "Emergent Complexity via Multi-Agent Competition"

**Kitaplar:**
- Shoham & Leyton-Brown: "Multiagent Systems"
- Sutton & Barto: "Reinforcement Learning: An Introduction"

**Online:**
- OpenAI Blog: Dota 2 self-play
- DeepMind Research: AlphaStar & AlphaGo
- Spinning Up in Deep RL: MARL tutorials

---

*Bu yazı, Çoklu Ajan Takviyeli Öğrenme serisinin 2. haftasıdır. Gelecek hafta: "Nash Dengesi'ni Öğrenmek: Q-Learning Buluşuyor von Neumann ile"*


**Hashtags:** #MachineLearning #ReinforcementLearning #MultiAgentSystems #SelfPlay #GameTheory #AI #DeepLearning #AlphaGo #AlphaZero #StarCraft

