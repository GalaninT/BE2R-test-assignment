def select_trajectory(generator, predictor, state, depth, command, N, H):
    """
    Выбирает лучшую траекторию из N кандидатов.
    
    Args:
        generator: обученный генератор траекторий
        predictor: обученная предсказательная ветвь
        state: текущее проприоцептивное состояние
        depth: изображения глубины
        command: команда скорости
        N: число кандидатов
        H: горизонт предсказания (шагов вперёд)
    
    Returns:
        best_trajectory: лучшая траектория
    """
    candidates = []
    for i in range(N):
        #Генерируем траекторию со случайным шумом
        tau = generator.sample(depth, command, noise=torch.randn(...))
        
        #Проигрываем H шагов вперёд через предсказатель
        score = 0
        state_sim = state
        for h in range(H):
            z_prop, z_contact = predictor(state_sim, tau[h])
            
            #Оценка
            score += evaluate(z_prop, z_contact, command)
            state_sim = z_prop  #бновлнение состояние
        
        candidates.append((score, tau))
    
    #Выбираем лучшую
    best_trajectory = max(candidates, key=lambda x: x[0])[1]
    return best_trajectory
