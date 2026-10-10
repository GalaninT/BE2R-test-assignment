class PredictiveBranch(nn.Module):
    def __init__(self):
        self.prop_encoder = MLP(prop_dim, latent_dim)
        self.contact_encoder = MLP(contact_dim, latent_dim)
        
        #Память
        self.gru = nn.GRU(latent_dim * 2, hidden_dim)
        
        #Предсказательные головы (обусловленные действием)
        self.prop_head = MLP(hidden_dim + action_dim, latent_dim)
        self.contact_head = MLP(hidden_dim + action_dim, latent_dim)
    
    def forward(self, prop_history, contact_history, action_history):
        #Кодируем историю
        z_prop = self.prop_encoder(prop_history)
        z_contact = self.contact_encoder(contact_history)
        
        #Память
        h = self.gru(torch.cat([z_prop, z_contact], dim=-1))
        
        #Предсказание следующего состояния, обусловленное действием
        z_prop_next = self.prop_head(torch.cat([h, action_history[-1]], dim=-1))
        z_contact_next = self.contact_head(torch.cat([h, action_history[-1]], dim=-1))
        
        return z_prop_next, z_contact_next
    
    def loss(self, z_prop_next, z_prop_target, z_contact_next, z_contact_target):
        L_pred = MSE(z_prop_next, z_prop_target) + MSE(z_contact_next, z_contact_target)
        L_sigreg = self.sigreg(z_prop_next) + self.sigreg(z_contact_next)
        return L_pred + 0.1 * L_sigreg
