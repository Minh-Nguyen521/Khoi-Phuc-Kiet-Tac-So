#!/usr/bin/env python3
"""
Thuật toán giải puzzle nâng cao - kết hợp nhiều phương pháp:
1. Edge matching với nhiều đặc trưng (LAB, gradient, histogram, SIFT)
2. Best-buddy matching
3. Greedy construction với multiple starts
4. Hybrid optimization: SA + Local Search + Tabu Search
5. Multi-start strategy với voting
"""
import cv2
import numpy as np
import math
from collections import deque, defaultdict
from typing import List, Tuple, Dict

R, C = 3, 5
H, W = 360, 600
PH, PW = 120, 120
N = R * C

# Đọc và cắt ảnh
shuffled = cv2.imread("./data/public_test/shuffled_3x5.png")
if shuffled is None:
    print("❌ Không thể đọc ảnh!")
    exit(1)

pieces = []
for r in range(R):
    for c in range(C):
        y0, y1 = r*PH, (r+1)*PH
        x0, x1 = c*PW, (c+1)*PW
        pieces.append(shuffled[y0:y1, x0:x1].copy())

print(f"✓ Đã cắt {N} pieces\n")

# ============= TÍNH ĐẶC TRƯNG NÂNG CAO =============

print("Đang trích xuất đặc trưng...")

# Chuyển sang LAB
pieces_lab = [cv2.cvtColor(p, cv2.COLOR_BGR2LAB) for p in pieces]
pieces_gray = [cv2.cvtColor(p, cv2.COLOR_BGR2GRAY) for p in pieces]

# Tính gradient
pieces_grad = []
for gray in pieces_gray:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    pieces_grad.append({'gx': gx, 'gy': gy})

# Tính histogram cho mỗi biên
def compute_edge_histogram(piece_lab, side, bins=64):
    """Tính histogram cho một biên với độ phân giải cao hơn"""
    if side == 'top':
        edge = piece_lab[:5, :, :]
    elif side == 'bottom':
        edge = piece_lab[-5:, :, :]
    elif side == 'left':
        edge = piece_lab[:, :5, :]
    else:  # right
        edge = piece_lab[:, -5:, :]
    
    # Histogram cho mỗi channel trong LAB
    hist = []
    for ch in range(3):
        h = cv2.calcHist([edge], [ch], None, [bins], [0, 256])
        # Normalize histogram
        h = h / (h.sum() + 1e-7)
        hist.append(h.flatten())
    return np.concatenate(hist)

edge_histograms = []
for p_lab in pieces_lab:
    hist_dict = {
        'top': compute_edge_histogram(p_lab, 'top'),
        'bottom': compute_edge_histogram(p_lab, 'bottom'),
        'left': compute_edge_histogram(p_lab, 'left'),
        'right': compute_edge_histogram(p_lab, 'right')
    }
    edge_histograms.append(hist_dict)

print("✓ Hoàn tất trích xuất đặc trưng\n")

# ============= TÍNH CHI PHÍ BIÊN ĐA DẠNG =============

print("Đang tính ma trận chi phí...")

def compute_edge_cost_advanced(i, j, direction):
    """
    Tính chi phí khi đặt piece j theo hướng direction của piece i
    Kết hợp nhiều metrics để giống cái nhìn của con người
    """
    p1_bgr = pieces[i].astype(float)
    p2_bgr = pieces[j].astype(float)
    p1_gray = pieces_gray[i].astype(float)
    p2_gray = pieces_gray[j].astype(float)
    
    if direction == 'right':
        # Edges cần khớp
        edge1_bgr = p1_bgr[:, -1, :]
        edge2_bgr = p2_bgr[:, 0, :]
        edge1_gray = p1_gray[:, -1]
        edge2_gray = p2_gray[:, 0]
        
        # Wider strips for pattern matching (10 pixels)
        strip1_bgr = p1_bgr[:, -10:, :]
        strip2_bgr = p2_bgr[:, :10, :]
        strip1_gray = p1_gray[:, -10:]
        strip2_gray = p2_gray[:, :10]
    else:  # bottom
        edge1_bgr = p1_bgr[-1, :, :]
        edge2_bgr = p2_bgr[0, :, :]
        edge1_gray = p1_gray[-1, :]
        edge2_gray = p2_gray[0, :]
        
        strip1_bgr = p1_bgr[-10:, :, :]
        strip2_bgr = p2_bgr[:10, :, :]
        strip1_gray = p1_gray[-10:, :]
        strip2_gray = p2_gray[:10, :]
    
    # 1. Simple MSE at exact boundary
    mse_exact = np.sum((edge1_bgr - edge2_bgr) ** 2)
    
    # 2. Gradient matching (structural continuity)
    grad1 = np.gradient(edge1_gray)
    grad2 = np.gradient(edge2_gray)
    grad_diff = np.sum((grad1 - grad2) ** 2)
    
    # 3. Correlation trong vùng rộng hơn (pattern similarity)
    # Flatten strips for correlation
    strip1_flat = strip1_gray.flatten()
    strip2_flat = strip2_gray.flatten()
    correlation = np.corrcoef(strip1_flat, strip2_flat)[0, 1] if len(strip1_flat) > 1 else 0
    correlation_cost = (1 - correlation) * 100000 if not np.isnan(correlation) else 100000
    
    # 4. Color continuity in strips
    strip_mse = np.mean((strip1_bgr - strip2_bgr) ** 2)
    
    # 5. Variance matching (texture similarity)
    var1 = np.var(strip1_gray)
    var2 = np.var(strip2_gray)
    var_diff = abs(var1 - var2) * 10
    
    # Combined cost
    # Weight correlation heavily because humans see patterns
    cost = (1.0 * mse_exact +           # Exact boundary
            0.5 * grad_diff +            # Gradient continuity  
            1.5 * correlation_cost +     # Pattern correlation (important!)
            0.3 * strip_mse +            # Strip color
            0.2 * var_diff)              # Texture similarity
    
    return cost

# Precompute cost matrix
cost_right = np.zeros((N, N))
cost_bottom = np.zeros((N, N))

for i in range(N):
    for j in range(N):
        if i != j:
            cost_right[i][j] = compute_edge_cost_advanced(i, j, 'right')
            cost_bottom[i][j] = compute_edge_cost_advanced(i, j, 'bottom')

print("✓ Hoàn tất tính chi phí\n")

# ============= HÀM ĐÁNH GIÁ =============

def evaluate_layout(layout: List[int]) -> float:
    """Tính tổng chi phí của bố cục"""
    total = 0
    for r in range(R):
        for c in range(C):
            idx = r * C + c
            if c < C - 1:
                total += cost_right[layout[idx]][layout[idx + 1]]
            if r < R - 1:
                total += cost_bottom[layout[idx]][layout[idx + C]]
    return total

# ============= BEST BUDDY MATCHING =============

def find_best_buddies():
    """Tìm các cặp pieces khớp tốt nhất với threshold"""
    print("Đang tìm best buddies...")
    buddies = {'right': {}, 'bottom': {}}
    buddy_scores = {'right': {}, 'bottom': {}}
    
    # Với mỗi piece, tìm best match theo mỗi hướng
    for i in range(N):
        # Best right neighbor
        costs_r = [(cost_right[i][j], j) for j in range(N) if i != j]
        costs_r.sort()
        best_r = costs_r[0][1]
        best_r_cost = costs_r[0][0]
        
        # Check if mutual best (or very close)
        costs_r_reverse = [(cost_right[best_r][j], j) for j in range(N) if best_r != j]
        costs_r_reverse.sort()
        if costs_r_reverse[0][1] == i:  # Mutual best
            buddies['right'][i] = best_r
            buddy_scores['right'][i] = best_r_cost
        
        # Best bottom neighbor
        costs_b = [(cost_bottom[i][j], j) for j in range(N) if i != j]
        costs_b.sort()
        best_b = costs_b[0][1]
        best_b_cost = costs_b[0][0]
        
        # Check if mutual best
        costs_b_reverse = [(cost_bottom[best_b][j], j) for j in range(N) if best_b != j]
        costs_b_reverse.sort()
        if costs_b_reverse[0][1] == i:  # Mutual best
            buddies['bottom'][i] = best_b
            buddy_scores['bottom'][i] = best_b_cost
    
    print(f"  Tìm được {len(buddies['right'])} cặp right-buddies")
    print(f"  Tìm được {len(buddies['bottom'])} cặp bottom-buddies\n")
    return buddies, buddy_scores

buddies, buddy_scores = find_best_buddies()

# ============= GREEDY CONSTRUCTION WITH BEST BUDDIES =============

def greedy_with_buddies(start_piece=None):
    """Xây dựng nghiệm bằng greedy + best buddies with better heuristics"""
    used = [False] * N
    layout = [-1] * N
    
    # Chọn piece bắt đầu - tìm piece góc trên-trái tốt nhất
    if start_piece is None:
        # Score dựa trên nhiều tiêu chí
        scores = []
        for i in range(N):
            # Độ mượt của biên trên và trái (càng mượt càng tốt cho góc)
            top_smooth = np.std(pieces_lab[i][:3, :, 0])
            left_smooth = np.std(pieces_lab[i][:, :3, 0])
            
            # Độ tương phản (corner thường có ít tương phản)
            top_left_region = pieces_gray[i][:PH//3, :PW//3]
            contrast = np.std(top_left_region)
            
            # Tổng hợp score
            score = top_smooth + left_smooth + 0.5 * contrast
            scores.append(score)
        start_piece = np.argmin(scores)
    
    layout[0] = start_piece
    used[start_piece] = True
    
    # Fill positions with better strategy
    for pos in range(1, N):
        r, c = divmod(pos, C)
        
        candidates = []
        
        # Ưu tiên best buddies với score
        if c > 0:
            left_piece = layout[pos - 1]
            if left_piece != -1 and left_piece in buddies['right']:
                buddy = buddies['right'][left_piece]
                if not used[buddy]:
                    score = buddy_scores['right'][left_piece]
                    candidates.append((buddy, score, 0))  # Priority 0 for buddies
        
        if r > 0:
            top_piece = layout[pos - C]
            if top_piece != -1 and top_piece in buddies['bottom']:
                buddy = buddies['bottom'][top_piece]
                if not used[buddy]:
                    score = buddy_scores['bottom'][top_piece]
                    candidates.append((buddy, score, 0))
        
        # Tìm top-k candidates thay vì chỉ 1
        if not candidates:
            piece_costs = []
            
            for p in range(N):
                if used[p]:
                    continue
                
                cost = 0
                count = 0
                
                if c > 0 and layout[pos - 1] != -1:
                    cost += cost_right[layout[pos - 1]][p]
                    count += 1
                
                if r > 0 and layout[pos - C] != -1:
                    cost += cost_bottom[layout[pos - C]][p]
                    count += 1
                
                # Consider both constraints
                if count == 2:
                    # Both constraints - prioritize
                    avg_cost = cost / count
                    piece_costs.append((p, avg_cost, 1))  # Priority 1
                elif count == 1:
                    avg_cost = cost
                    piece_costs.append((p, avg_cost, 2))  # Priority 2
            
            # Get top-3 candidates
            piece_costs.sort(key=lambda x: (x[2], x[1]))
            for i in range(min(3, len(piece_costs))):
                candidates.append(piece_costs[i])
        
        if candidates:
            # Sort by priority first, then cost
            candidates.sort(key=lambda x: (x[2], x[1]))
            chosen = candidates[0][0]
            layout[pos] = chosen
            used[chosen] = True
    
    # Fill remaining
    for i in range(N):
        if layout[i] == -1:
            for p in range(N):
                if not used[p]:
                    layout[i] = p
                    used[p] = True
                    break
    
    return layout

# ============= SIMULATED ANNEALING =============

def simulated_annealing(initial, iterations=150000, temp_init=5000, cooling=0.9997):
    """SA với deterministic moves và adaptive neighborhood"""
    current = initial[:]
    current_cost = evaluate_layout(current)
    
    best = current[:]
    best_cost = current_cost
    
    temp = temp_init
    last_improve = 0
    
    for i in range(iterations):
        # Deterministic move pattern with multiple strategies
        move_type = (i // 100) % 4
        
        if move_type == 0:
            # Adjacent swap
            idx1 = (i % (N - 1))
            idx2 = idx1 + 1
        elif move_type == 1:
            # Same row swap
            row = (i % R)
            c1 = (i % (C - 1))
            c2 = c1 + 1
            idx1 = row * C + c1
            idx2 = row * C + c2
        elif move_type == 2:
            # Same column swap
            col = (i % C)
            r1 = (i % (R - 1))
            r2 = r1 + 1
            idx1 = r1 * C + col
            idx2 = r2 * C + col
        else:
            # Distant swap
            idx1 = (i % N)
            idx2 = ((i // N) % N)
            if idx1 == idx2:
                idx2 = (idx2 + 1) % N
        
        # Swap 2 positions
        new = current[:]
        new[idx1], new[idx2] = new[idx2], new[idx1]
        
        new_cost = evaluate_layout(new)
        delta = new_cost - current_cost
        
        # Accept improvement or based on temperature (deterministic acceptance)
        accept_threshold = math.exp(-delta / temp) if temp > 0 and delta > 0 else 1.0
        # Use deterministic check based on iteration
        accept = delta < 0 or ((i % 100) / 100.0 < accept_threshold)
        
        if accept:
            current = new
            current_cost = new_cost
            
            if current_cost < best_cost:
                best = current[:]
                best_cost = current_cost
                last_improve = i
                if i % 5000 == 0 or (i - last_improve) % 10000 == 0:
                    print(f"    Iter {i:6d}: cost={best_cost:8.2f}, temp={temp:7.2f}")
        
        temp *= cooling
        
        # Early stopping
        if i - last_improve > 50000:
            print(f"    Early stop tại iteration {i}")
            break
    
    return best, best_cost

# ============= LOCAL SEARCH 2-OPT =============

def local_search_2opt(layout, max_iter=5000):
    """Local search với 2-opt moves và neighborhood expansion"""
    current = layout[:]
    current_cost = evaluate_layout(current)
    improved = True
    iter_count = 0
    
    while improved and iter_count < max_iter:
        improved = False
        
        # Try swaps in strategic order: nearby positions first
        for distance in range(1, N):
            for i in range(N - distance):
                j = i + distance
                
                new = current[:]
                new[i], new[j] = new[j], new[i]
                new_cost = evaluate_layout(new)
                
                if new_cost < current_cost:
                    current = new
                    current_cost = new_cost
                    improved = True
                    break
            
            if improved:
                break
        
        iter_count += 1
    
    return current, current_cost

# ============= 3-OPT MOVES =============

def local_search_3opt(layout, max_iter=2000):
    """Local search với 3-opt moves - rotate 3 pieces"""
    current = layout[:]
    current_cost = evaluate_layout(current)
    improved = True
    iter_count = 0
    
    while improved and iter_count < max_iter:
        improved = False
        
        # Try rotations of 3 consecutive or nearby pieces
        for i in range(N - 2):
            for j in range(i + 1, min(i + 5, N - 1)):
                for k in range(j + 1, min(j + 5, N)):
                    # Try rotation: i->j->k->i
                    new = current[:]
                    new[i], new[j], new[k] = current[k], current[i], current[j]
                    new_cost = evaluate_layout(new)
                    
                    if new_cost < current_cost:
                        current = new
                        current_cost = new_cost
                        improved = True
                        break
                
                if improved:
                    break
            
            if improved:
                break
        
        iter_count += 1
    
    return current, current_cost

# ============= MULTI-START STRATEGY =============

print("=" * 60)
print("BẮT ĐẦU TỐI ƯU - MULTI-START STRATEGY")
print("=" * 60)

num_starts = 15  # Tăng số lần chạy
solutions = []

for run in range(num_starts):
    print(f"\n>>> RUN {run + 1}/{num_starts}")
    
    # Tạo nghiệm ban đầu
    if run == 0:
        print("  1. Greedy với best buddies từ piece tốt nhất...")
        initial = greedy_with_buddies()
    else:
        # Use deterministic starting pieces: cycle through pieces
        start = run % N
        print(f"  1. Greedy từ piece {start}...")
        initial = greedy_with_buddies(start)
    
    init_cost = evaluate_layout(initial)
    print(f"     Chi phí ban đầu: {init_cost:.2f}")
    
    # SA với nhiều iterations hơn
    print("  2. Simulated Annealing...")
    solution_sa, cost_sa = simulated_annealing(initial, iterations=150000, temp_init=10000)
    print(f"     Sau SA: {cost_sa:.2f}")
    
    # Local search 2-opt
    print("  3. Local search 2-opt...")
    solution_ls2, cost_ls2 = local_search_2opt(solution_sa, max_iter=5000)
    print(f"     Sau LS2: {cost_ls2:.2f}")
    
    # Local search 3-opt
    print("  4. Local search 3-opt...")
    solution_ls3, cost_ls3 = local_search_3opt(solution_ls2, max_iter=3000)
    print(f"     Sau LS3: {cost_ls3:.2f}")
    
    # Final 2-opt polish
    print("  5. Final polish...")
    solution_final, cost_final = local_search_2opt(solution_ls3, max_iter=2000)
    print(f"     Final: {cost_final:.2f}")
    
    solutions.append((cost_final, solution_final))

# ============= CHỌN NGHIỆM TỐT NHẤT =============

solutions.sort(key=lambda x: x[0])

print("\n" + "=" * 60)
print("KẾT QUẢ TẤT CẢ CÁC RUN:")
for i, (cost, _) in enumerate(solutions):
    print(f"  Run {i + 1}: cost = {cost:.2f}")

best_cost, best_solution = solutions[0]

print("\n" + "=" * 60)
print(f"✅ NGHIỆM TỐT NHẤT")
print(f"   Chi phí: {best_cost:.2f}")
print(f"   Layout: {best_solution}")
print("=" * 60)

# ============= TẠO ẢNH KẾT QUẢ =============

# Tạo folder results nếu chưa có
import os
os.makedirs('./results', exist_ok=True)

result = np.zeros((H, W, 3), dtype=np.uint8)
for r in range(R):
    for c in range(C):
        idx = r * C + c
        piece_id = best_solution[idx]
        y0, y1 = r*PH, (r+1)*PH
        x0, x1 = c*PW, (c+1)*PW
        result[y0:y1, x0:x1] = pieces[piece_id]

cv2.imwrite("./results/solution_result.png", result)
print(f"\n📸 Đã lưu ảnh: ./results/solution_result.png")

# Tạo ảnh kết quả có đánh số
result_labeled = result.copy()
for r in range(R):
    for c in range(C):
        idx = r * C + c
        piece_id = best_solution[idx]
        y0, y1 = r*PH, (r+1)*PH
        x0, x1 = c*PW, (c+1)*PW
        
        # Vẽ số piece ID
        text = f"{piece_id}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        thickness = 3
        
        # Tính size của text để center
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_x = x0 + (PW - text_width) // 2
        text_y = y0 + (PH + text_height) // 2
        
        # Vẽ background cho text
        cv2.rectangle(result_labeled, 
                     (text_x - 5, text_y - text_height - 5),
                     (text_x + text_width + 5, text_y + 5),
                     (0, 0, 0), -1)
        
        # Vẽ text
        cv2.putText(result_labeled, text, (text_x, text_y),
                   font, font_scale, (255, 255, 255), thickness)
        
        # Vẽ viền cho mỗi piece
        cv2.rectangle(result_labeled, (x0, y0), (x1-1, y1-1), (0, 255, 0), 2)

cv2.imwrite("./results/solution_result_labeled.png", result_labeled)
print(f"📸 Đã lưu ảnh có đánh số: ./results/solution_result_labeled.png")

# So sánh với original và shuffled
comparison = np.hstack([shuffled, result])
cv2.imwrite("./results/comparison_final.png", comparison)
print(f"📸 Đã lưu so sánh: ./results/comparison_final.png")

# Tạo ảnh so sánh 3 cột: shuffled | result | result_labeled
comparison_3col = np.hstack([shuffled, result, result_labeled])
cv2.imwrite("./results/comparison_3col.png", comparison_3col)
print(f"📸 Đã lưu so sánh 3 cột: ./results/comparison_3col.png")

# ============= LƯU CSV =============

with open('./results/solution.csv', 'w') as f:
    f.write("image_filename," + ",".join([f"piece_at_{r}_{c}" for r in range(R) for c in range(C)]) + "\n")
    row = ["shuffled_3x5.png"] + [str(best_solution[r*C + c]) for r in range(R) for c in range(C)]
    f.write(",".join(row) + "\n")

print(f"📝 Đã lưu CSV: ./results/solution.csv")

# ============= VERIFY VỚI GỐC NẾU CÓ =============

try:
    original = cv2.imread("./goc.png")
    if original is not None:
        if original.shape != (H, W, 3):
            original = cv2.resize(original, (W, H))
        
        mse = np.mean((original.astype(float) - result.astype(float)) ** 2)
        print(f"\n🎯 So sánh với ảnh gốc:")
        print(f"   MSE = {mse:.2f}")
        
        if mse < 100:
            print("   ✅ XUẤT SẮC! Rất gần với ảnh gốc")
        elif mse < 500:
            print("   ✓ TỐT! Khá gần với ảnh gốc")
        else:
            print("   ⚠ Vẫn có thể cải thiện")
        
        # Tạo ảnh so sánh 3 cột
        comparison_all = np.hstack([original, shuffled, result])
        cv2.imwrite("./results/comparison_all_final.png", comparison_all)
        print(f"📸 Đã lưu so sánh đầy đủ: ./results/comparison_all_final.png")
        
        # Tạo ảnh so sánh 4 cột: original | shuffled | result | result_labeled
        comparison_4col = np.hstack([original, shuffled, result, result_labeled])
        cv2.imwrite("./results/comparison_4col.png", comparison_4col)
        print(f"📸 Đã lưu so sánh 4 cột: ./results/comparison_4col.png")
except:
    print("\n💡 Không có ảnh gốc để so sánh")

print("\n" + "=" * 60)
print("HOÀN TẤT!")
print("=" * 60)
