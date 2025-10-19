import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import math
from collections import deque
import matplotlib.pyplot as plt

# GPU kullan varsa
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 Kullanılan cihaz: {device}")


class TicTacToe:
    """Tic-Tac-Toe oyun ortamı - AlphaZero uyumlu"""
    
    def __init__(self):
        self.row_count = 3
        self.column_count = 3
        self.action_size = 9  # 3x3 = 9 hücre
        
    def __repr__(self):
        return "TicTacToe"
        
    def get_initial_state(self):
        """Boş tahta döndür"""
        return np.zeros((self.row_count, self.column_count), dtype=np.float32)
    
    def get_next_state(self, state, action, player):
        """Hamle yap ve yeni state döndür"""
        state = state.copy()
        row = action // self.column_count
        col = action % self.column_count
        state[row, col] = player
        return state
    
    def get_valid_moves(self, state):
        """Geçerli hamleleri binary mask olarak döndür"""
        return (state.reshape(-1) == 0).astype(np.uint8)
    
    def check_win(self, state, action):
        """Kazanan var mı kontrol et"""
        if action is None:
            return False
        
        row = action // self.column_count
        col = action % self.column_count
        player = state[row, col]
        
        # Satır kontrolü
        if np.sum(state[row, :]) == player * self.column_count:
            return True
        
        # Sütun kontrolü
        if np.sum(state[:, col]) == player * self.row_count:
            return True
        
        # Ana çapraz
        if np.sum(np.diag(state)) == player * self.row_count:
            return True
        
        # Ters çapraz
        if np.sum(np.diag(np.flip(state, axis=0))) == player * self.row_count:
            return True
        
        return False
    
    def get_value_and_terminated(self, state, action):
        """Oyun bitti mi ve değeri ne? (1: kazandı, 0: devam/berabere)"""
        if self.check_win(state, action):
            return 1, True
        if np.sum(self.get_valid_moves(state)) == 0:
            return 0, True  # Beraberlik
        return 0, False  # Oyun devam ediyor
    
    def get_opponent(self, player):
        """Rakip oyuncu döndür"""
        return -player
    
    def get_opponent_value(self, value):
        """Rakip perspektifinden değer"""
        return -value
    
    def change_perspective(self, state, player):
        """State'i current player perspektifine çevir"""
        return state * player
    
    def get_encoded_state(self, state):
        """Neural network için encoded state (3 channel: -1, 0, +1)"""
        encoded_state = np.stack([
            (state == -1).astype(np.float32),
            (state == 0).astype(np.float32),
            (state == 1).astype(np.float32)
        ])
        
        # Batch için shape düzeltme
        if len(state.shape) == 3:
            encoded_state = np.swapaxes(encoded_state, 0, 1)
        
        return encoded_state
    
    def render(self, state):
        """Tahtayı görselleştir"""
        symbols = {0: '.', 1: 'X', -1: 'O'}
        print("\n  0 1 2")
        for i in range(3):
            print(f"{i} " + ' '.join(symbols[state[i, j]] for j in range(3)))
        print()


class ResNet(nn.Module):
    """AlphaZero tarzı ResNet - Policy-Value Network"""
    
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
        
        # Policy head
        self.policyHead = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * game.row_count * game.column_count, game.action_size)
        )
        
        # Value head
        self.valueHead = nn.Sequential(
            nn.Conv2d(num_hidden, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * game.row_count * game.column_count, 1),
            nn.Tanh()
        )
        
        self.to(device)
        
    def forward(self, x):
        x = self.startBlock(x)
        for resBlock in self.backBone:
            x = resBlock(x)
        
        policy = self.policyHead(x)
        value = self.valueHead(x)
        
        return policy, value


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
        x = F.relu(x)
        return x


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
        """Bu node expand edilmiş mi?"""
        return len(self.children) > 0
    
    def select(self):
        """En yüksek UCB skoruna sahip child'ı seç"""
        best_child = None
        best_ucb = -np.inf
        
        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb
        
        return best_child
    
    def get_ucb(self, child):
        """UCB (Upper Confidence Bound) skoru - AlphaZero PUCT formülü"""
        if child.visit_count == 0:
            q_value = 0
        else:
            # Q-value: ortalama değer [-1, 1] aralığında
            q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
        
        # U-value: exploration bonus
        u_value = (self.args['C'] * child.prior * 
                   math.sqrt(self.visit_count) / (1 + child.visit_count))
        
        return q_value + u_value
    
    def expand(self, policy):
        """Node'u expand et - her valid action için child oluştur"""
        for action, prob in enumerate(policy):
            if prob > 0:
                child_state = self.state.copy()
                child_state = self.game.get_next_state(child_state, action, 1)
                child_state = self.game.change_perspective(child_state, player=-1)
                
                child = Node(self.game, self.args, child_state, self, action, prob)
                self.children.append(child)
    
    def backpropagate(self, value):
        """Value'yu tree'de yukarı doğru yay"""
        self.value_sum += value
        self.visit_count += 1
        
        value = self.game.get_opponent_value(value)
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
        # Root node oluştur
        root = Node(self.game, self.args, state, visit_count=1)
        
        # Neural network ile initial policy al
        policy, _ = self.model(
            torch.tensor(self.game.get_encoded_state(state), device=device).unsqueeze(0)
        )
        policy = torch.softmax(policy, dim=1).squeeze(0).cpu().numpy()
        
        # Dirichlet noise ekle (exploration için)
        policy = ((1 - self.args['dirichlet_epsilon']) * policy + 
                  self.args['dirichlet_epsilon'] * 
                  np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size))
        
        # Invalid moves'u mask'le
        valid_moves = self.game.get_valid_moves(state)
        policy *= valid_moves
        policy /= np.sum(policy)
        
        # Root'u expand et
        root.expand(policy)
        
        # MCTS simulations
        for _ in range(self.args['num_searches']):
            node = root
            
            # Selection: leaf node'a kadar in
            while node.is_fully_expanded():
                node = node.select()
            
            # Evaluation
            value, is_terminal = self.game.get_value_and_terminated(
                node.state, node.action_taken
            )
            value = self.game.get_opponent_value(value)
            
            # Expansion (eğer terminal değilse)
            if not is_terminal:
                policy, value = self.model(
                    torch.tensor(self.game.get_encoded_state(node.state), 
                               device=device).unsqueeze(0)
                )
                policy = torch.softmax(policy, dim=1).squeeze(0).cpu().numpy()
                
                # Invalid moves'u mask'le
                valid_moves = self.game.get_valid_moves(node.state)
                policy *= valid_moves
                policy /= np.sum(policy)
                
                value = value.item()
                
                # Expand
                node.expand(policy)
            
            # Backpropagation
            node.backpropagate(value)
        
        # Visit count'lara göre action probabilities hesapla
        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= np.sum(action_probs)
        
        return action_probs


class AlphaZero:
    """AlphaZero Algorithm"""
    
    def __init__(self, model, optimizer, game, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTS(game, args, model)
    
    def selfPlay(self):
        """Bir self-play oyunu oyna ve training data topla"""
        memory = []
        player = 1
        state = self.game.get_initial_state()
        
        while True:
            # Current player perspektifinden state
            neutral_state = self.game.change_perspective(state, player)
            
            # MCTS ile action probabilities
            action_probs = self.mcts.search(neutral_state)
            
            # Training data'ya ekle
            memory.append((neutral_state, action_probs, player))
            
            # Temperature-based action selection
            temperature_action_probs = action_probs ** (1 / self.args['temperature'])
            temperature_action_probs /= np.sum(temperature_action_probs)
            action = np.random.choice(self.game.action_size, p=temperature_action_probs)
            
            # State'i güncelle
            state = self.game.get_next_state(state, action, player)
            
            # Terminal kontrolü
            value, is_terminal = self.game.get_value_and_terminated(state, action)
            
            if is_terminal:
                # Gerçek kazananı belirle (perspektif değişiminden ÖNCE)
                real_winner = player if value == 1 else (0 if value == 0 else -player)
                
                # Her pozisyona outcome ata
                return_memory = []
                for hist_neutral_state, hist_action_probs, hist_player in memory:
                    hist_outcome = (value if hist_player == player 
                                  else self.game.get_opponent_value(value))
                    return_memory.append((
                        self.game.get_encoded_state(hist_neutral_state),
                        hist_action_probs,
                        hist_outcome
                    ))
                return return_memory, real_winner  # Gerçek kazananı da döndür
            
            # Sıra diğer oyuncuya geçer
            player = self.game.get_opponent(player)
    
    def train(self, memory):
        """Neural network'ü eğit"""
        random.shuffle(memory)
        
        for batchIdx in range(0, len(memory), self.args['batch_size']):
            sample = memory[batchIdx:min(len(memory), batchIdx + self.args['batch_size'])]
            
            state, policy_targets, value_targets = zip(*sample)
            
            state = np.array(state)
            policy_targets = np.array(policy_targets)
            value_targets = np.array(value_targets).reshape(-1, 1)
            
            # Tensors
            state = torch.tensor(state, dtype=torch.float32, device=device)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=device)
            
            # Forward pass
            out_policy, out_value = self.model(state)
            
            # Loss
            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        return loss.item()
    
    def learn(self):
        """Ana eğitim loop'u"""
        stats = {
            'iteration': [],
            'x_wins': [],
            'o_wins': [],
            'draws': [],
            'avg_loss': []
        }
        
        print("=" * 70)
        print("🌳 ALPHAZERO SELF-PLAY TRAINING")
        print("=" * 70)
        print(f"İterasyonlar: {self.args['num_iterations']}")
        print(f"Her iterasyonda self-play oyunlar: {self.args['num_selfPlay_iterations']}")
        print(f"MCTS simulations: {self.args['num_searches']}")
        print("=" * 70)
        
        for iteration in range(self.args['num_iterations']):
            print(f"\n🔄 İterasyon {iteration + 1}/{self.args['num_iterations']}")
            
            # Self-play phase
            memory = []
            self.model.eval()
            
            x_wins = 0
            o_wins = 0
            draws = 0
            
            for sp_iter in range(self.args['num_selfPlay_iterations']):
                game_memory, real_winner = self.selfPlay()
                memory.extend(game_memory)
                
                # Gerçek kazananı say (real_winner: 1=X, -1=O, 0=Draw)
                if real_winner == 1:
                    x_wins += 1
                elif real_winner == -1:
                    o_wins += 1
                else:
                    draws += 1
                
                if (sp_iter + 1) % 20 == 0:
                    print(f"  Self-play: {sp_iter + 1}/{self.args['num_selfPlay_iterations']}")
            
            total_games = self.args['num_selfPlay_iterations']
            print(f"  📊 Sonuçlar: X kazandı={x_wins} ({100*x_wins/total_games:.1f}%), "
                  f"O kazandı={o_wins} ({100*o_wins/total_games:.1f}%), "
                  f"Berabere={draws} ({100*draws/total_games:.1f}%)")
            print(f"  💾 Toplanan training data: {len(memory)} pozisyon")
            
            # Training phase
            self.model.train()
            total_loss = 0
            
            for epoch in range(self.args['num_epochs']):
                loss = self.train(memory)
                total_loss += loss
            
            avg_loss = total_loss / self.args['num_epochs']
            print(f"  📉 Ortalama Loss: {avg_loss:.4f}")
            
            # Stats kaydet
            stats['iteration'].append(iteration + 1)
            stats['x_wins'].append(100 * x_wins / total_games)
            stats['o_wins'].append(100 * o_wins / total_games)
            stats['draws'].append(100 * draws / total_games)
            stats['avg_loss'].append(avg_loss)
            
            # Checkpoint kaydet
            if (iteration + 1) % 5 == 0 or iteration == self.args['num_iterations'] - 1:
                torch.save(self.model.state_dict(), 
                          f'alphazero_tictactoe_iter{iteration+1}.pth')
                print(f"  💾 Checkpoint kaydedildi")
        
        print("\n" + "=" * 70)
        print("✅ EĞİTİM TAMAMLANDI!")
        print("=" * 70)
        
        return stats


def play_against_alphazero(model, game, args, human_player=1):
    """AlphaZero'ya karşı oyna"""
    mcts = MCTS(game, args, model)
    model.eval()
    
    state = game.get_initial_state()
    player = 1
    
    print("\n" + "=" * 70)
    print("🌳 İnsan vs AlphaZero (MCTS + Neural Network)")
    print("=" * 70)
    print("Hücreler: 0-8 (satır * 3 + sütun)")
    print("Layout: 0 1 2")
    print("        3 4 5")
    print("        6 7 8")
    print("=" * 70)
    
    game.render(state)
    
    while True:
        if player == human_player:
            # İnsan hamle
            valid_moves = game.get_valid_moves(state)
            valid_actions = [i for i in range(9) if valid_moves[i] == 1]
            print(f"Geçerli hamleler: {valid_actions}")
            
            while True:
                try:
                    action = int(input("👤 Hamleniz (0-8): "))
                    if action in valid_actions:
                        break
                    print("❌ Geçersiz hamle!")
                except ValueError:
                    print("❌ Lütfen 0-8 arası sayı girin")
        else:
            # AlphaZero hamle
            print("🌳 AlphaZero düşünüyor (MCTS çalışıyor)...")
            neutral_state = game.change_perspective(state, player)
            
            # MCTS search (greedy)
            args_temp = args.copy()
            args_temp['temperature'] = 0
            args_temp['dirichlet_epsilon'] = 0  # No exploration
            
            mcts_temp = MCTS(game, args_temp, model)
            action_probs = mcts_temp.search(neutral_state)
            action = np.argmax(action_probs)
            
            print(f"🌳 AlphaZero hamle: {action}")
        
        # Hamleyi yap
        state = game.get_next_state(state, action, player)
        game.render(state)
        
        # Terminal kontrolü
        value, is_terminal = game.get_value_and_terminated(state, action)
        
        if is_terminal:
            print("=" * 70)
            if value == 1:
                winner = "İnsan" if player == human_player else "AlphaZero"
                print(f"🎉 {winner} KAZANDI!")
            else:
                print("🤝 BERABERE!")
            print("=" * 70)
            break
        
        # Sıra değiştir
        player = game.get_opponent(player)


# Ana eğitim ve test
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🌳 ALPHAZERO TIC-TAC-TOE (Seviye 2)")
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
        'batch_size': 64,                # Batch size
        'temperature': 1.25,             # Temperature for action selection
        'dirichlet_epsilon': 0.25,       # Dirichlet noise weight
        'dirichlet_alpha': 0.3           # Dirichlet alpha parameter
    }
    
    # Eğitim
    print("\n🎓 Eğitim başlıyor...")
    alphazero = AlphaZero(model, optimizer, game, args)
    stats = alphazero.learn()
    
    # Sonuçları görselleştir
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Win rates
    axes[0, 0].plot(stats['iteration'], stats['x_wins'], 
                    'o-', label='X Kazandı %', color='#e74c3c', linewidth=2, markersize=6)
    axes[0, 0].plot(stats['iteration'], stats['o_wins'], 
                    's-', label='O Kazandı %', color='#3498db', linewidth=2, markersize=6)
    axes[0, 0].plot(stats['iteration'], stats['draws'], 
                    '^-', label='Berabere %', color='#2ecc71', linewidth=2, markersize=6)
    axes[0, 0].set_xlabel('İterasyon', fontweight='bold', fontsize=11)
    axes[0, 0].set_ylabel('Yüzde (%)', fontweight='bold', fontsize=11)
    axes[0, 0].set_title('Oyun Sonuçları (Gerçek Kazananlar)', fontweight='bold', fontsize=12)
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss
    axes[0, 1].plot(stats['iteration'], stats['avg_loss'], 
                    linewidth=2.5, color='#9b59b6', marker='o', markersize=5)
    axes[0, 1].set_xlabel('İterasyon', fontweight='bold', fontsize=11)
    axes[0, 1].set_ylabel('Loss (Kayıp)', fontweight='bold', fontsize=11)
    axes[0, 1].set_title('Eğitim Loss Değeri', fontweight='bold', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Draw convergence
    axes[1, 0].plot(stats['iteration'], stats['draws'], 
                    linewidth=3, color='#2ecc71', marker='o', markersize=7)
    axes[1, 0].axhline(y=70, color='red', linestyle='--', 
                       label='Hedef: ~70-80% (Optimal Oyun)', linewidth=2)
    axes[1, 0].set_xlabel('İterasyon', fontweight='bold', fontsize=11)
    axes[1, 0].set_ylabel('Beraberlik %', fontweight='bold', fontsize=11)
    axes[1, 0].set_title('Optimal Oyuna Yakınsama', fontweight='bold', fontsize=12)
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Summary text
    final_draws = stats['draws'][-1]
    final_x = stats['x_wins'][-1]
    final_o = stats['o_wins'][-1]
    final_loss = stats['avg_loss'][-1]
    axes[1, 1].text(0.5, 0.7, f'📊 Final İstatistikler', 
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
                    ha='center', va='center', fontsize=11, style='italic', color='#34495e')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.suptitle('🌳 AlphaZero Tic-Tac-Toe Eğitim Sonuçları', 
                 fontsize=14, fontweight='bold', y=1.00)
    plt.savefig('alphazero_tictactoe_results.png', dpi=150, bbox_inches='tight')
    print("\n📊 Grafik kaydedildi: alphazero_tictactoe_results.png")
    
    # İnsan vs AlphaZero
    print("\n" + "=" * 70)
    play_choice = input("AlphaZero'ya karşı oynamak ister misiniz? (e/h): ")
    if play_choice.lower() == 'e':
        player_choice = input("X (ilk) mi O (ikinci) mi? (x/o): ")
        human_player = 1 if player_choice.lower() == 'x' else -1
        
        play_against_alphazero(model, game, args, human_player)
        
        while True:
            again = input("\nBir daha oynamak ister misiniz? (e/h): ")
            if again.lower() == 'e':
                play_against_alphazero(model, game, args, human_player)
            else:
                print("\n🌳 Görüşmek üzere! AlphaZero rules! 🚀")
                break
    else:
        print("\n👋 Görüşmek üzere!")