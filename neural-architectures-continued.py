context)
        return output

class CrossModalTransformer(nn.Module):
    def __init__(
        self,
        dim_market: int = 128,
        dim_molecule: int = 256,
        dim_sensor: int = 64,
        nhead: int = 8,
        num_layers: int = 6
    ):
        super().__init__()
        self.dim_market = dim_market
        self.dim_molecule = dim_molecule
        self.dim_sensor = dim_sensor
        self.total_dim = dim_market + dim_molecule + dim_sensor
        
        self.market_encoder = nn.Linear(dim_market, dim_market)
        self.molecule_encoder = nn.Linear(dim_molecule, dim_molecule)
        self.sensor_encoder = nn.Linear(dim_sensor, dim_sensor)
        
        self.pos_encoder = PositionalEncoding(self.total_dim)
        
        encoder_layer = TransformerEncoderLayer(
            d_model=self.total_dim,
            nhead=nhead,
            dim_feedforward=4*self.total_dim,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers)
        
        self.market_decoder = nn.Linear(self.total_dim, dim_market)
        self.molecule_decoder = nn.Linear(self.total_dim, dim_molecule)
        self.sensor_decoder = nn.Linear(self.total_dim, dim_sensor)
        
    def forward(
        self,
        market: torch.Tensor,
        molecule: torch.Tensor,
        sensor: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Encode individual modalities
        market_encoded = self.market_encoder(market)
        molecule_encoded = self.molecule_encoder(molecule)
        sensor_encoded = self.sensor_encoder(sensor)
        
        # Combine encodings
        combined = torch.cat([market_encoded, molecule_encoded, sensor_encoded], dim=-1)
        
        # Add positional encoding
        combined = self.pos_encoder(combined)
        
        # Transform through layers
        transformed = self.transformer(combined, mask)
        
        # Decode back to individual modalities
        market_out = self.market_decoder(transformed[..., :self.dim_market])
        molecule_out = self.molecule_decoder(transformed[..., self.dim_market:self.dim_market+self.dim_molecule])
        sensor_out = self.sensor_decoder(transformed[..., -self.dim_sensor:])
        
        return market_out, molecule_out, sensor_out

class GatedFusion(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.fc1 = nn.Linear(dim_in, dim_out)
        self.fc2 = nn.Linear(dim_in, dim_out)
        self.gate = nn.Linear(dim_in, dim_out)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.fc1(x)
        h2 = self.fc2(x)
        g = torch.sigmoid(self.gate(x))
        return h1 * g + h2 * (1 - g)

class CrossDomainPolicy(nn.Module):
    def __init__(
        self,
        market_dim: int = 128,
        molecule_dim: int = 256,
        sensor_dim: int = 64,
        hidden_dim: int = 512,
        num_actions: int = 64
    ):
        super().__init__()
        self.market_encoder = nn.Sequential(
            nn.Linear(market_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            GatedFusion(hidden_dim, hidden_dim)
        )
        
        self.molecule_encoder = nn.Sequential(
            nn.Linear(molecule_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            GatedFusion(hidden_dim, hidden_dim)
        )
        
        self.sensor_encoder = nn.Sequential(
            nn.Linear(sensor_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            GatedFusion(hidden_dim, hidden_dim)
        )
        
        self.cross_attention = MultiModalAttention(
            hidden_dim, hidden_dim, hidden_dim
        )
        
        self.policy_head = nn.Sequential(
            nn.Linear(3 * hidden_dim, 2 * hidden_dim),
            nn.LayerNorm(2 * hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * hidden_dim, num_actions)
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(
        self,
        market_state: torch.Tensor,
        molecule_state: torch.Tensor,
        sensor_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Encode states
        market_encoded = self.market_encoder(market_state)
        molecule_encoded = self.molecule_encoder(molecule_state)
        sensor_encoded = self.sensor_encoder(sensor_state)
        
        # Cross-modal attention
        combined = self.cross_attention(
            market_encoded,
            molecule_encoded,
            sensor_encoded
        )
        
        # Policy and value outputs
        action_logits = self.policy_head(combined)
        value = self.value_head(combined)
        
        return action_logits, value

class CrossDomainPredictor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        sequence_length: int,
        num_heads: int = 8
    ):
        super().__init__()
        self.sequence_length = sequence_length
        
        self.input_embedding = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=4*hidden_dim,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=6
        )
        
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Input shape: (batch_size, sequence_length, input_dim)
        embedded = self.input_embedding(x)
        embedded = self.pos_encoder(embedded)
        
        transformed = self.transformer(embedded, mask)
        predictions = self.prediction_head(transformed)
        
        return predictions

class HierarchicalPredictor(nn.Module):
    def __init__(
        self,
        market_dim: int = 128,
        molecule_dim: int = 256,
        sensor_dim: int = 64,
        hidden_dim: int = 512,
        sequence_length: int = 100
    ):
        super().__init__()
        
        self.market_predictor = CrossDomainPredictor(
            market_dim, hidden_dim, sequence_length
        )
        self.molecule_predictor = CrossDomainPredictor(
            molecule_dim, hidden_dim, sequence_length
        )
        self.sensor_predictor = CrossDomainPredictor(
            sensor_dim, hidden_dim, sequence_length
        )
        
        self.cross_modal = CrossModalTransformer(
            dim_market=market_dim,
            dim_molecule=molecule_dim,
            dim_sensor=sensor_dim
        )
        
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(market_dim + molecule_dim + sensor_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # Uncertainty for each domain
        )
        
    def forward(
        self,
        market_history: torch.Tensor,
        molecule_history: torch.Tensor,
        sensor_history: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Individual domain predictions
        market_pred = self.market_predictor(market_history)
        molecule_pred = self.molecule_predictor(molecule_history)
        sensor_pred = self.sensor_predictor(sensor_history)
        
        # Cross-modal refinement
        market_refined, molecule_refined, sensor_refined = self.cross_modal(
            market_pred, molecule_pred, sensor_pred
        )
        
        # Uncertainty estimation
        combined = torch.cat([
            market_refined[:, -1],
            molecule_refined[:, -1],
            sensor_refined[:, -1]
        ], dim=-1)
        uncertainties = self.uncertainty_estimator(combined)
        
        return (
            market_refined,
            molecule_refined,
            sensor_refined,
            uncertainties
        )
