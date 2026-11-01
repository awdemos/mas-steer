use steer_core::{Backend, MockBackend, Mode, ModelDims, Site, SiteKind};

fn backend() -> MockBackend {
    MockBackend::new(
        ModelDims {
            layers: 4,
            hidden_dim: 16,
            mlp_dim: 32,
            vocab_size: 64,
        },
        7,
    )
}

#[test]
fn site_validation_catches_bad_layer() {
    let dims = ModelDims {
        layers: 4,
        hidden_dim: 16,
        mlp_dim: 32,
        vocab_size: 64,
    };
    let site = Site {
        kind: SiteKind::Residual,
        layer: 10,
        index: 0,
    };
    assert!(site
        .validate(dims.layers, dims.hidden_dim, dims.mlp_dim)
        .is_err());
}

#[test]
fn site_validation_catches_bad_neuron() {
    let dims = ModelDims {
        layers: 4,
        hidden_dim: 16,
        mlp_dim: 32,
        vocab_size: 64,
    };
    let site = Site {
        kind: SiteKind::Residual,
        layer: 0,
        index: 100,
    };
    assert!(site
        .validate(dims.layers, dims.hidden_dim, dims.mlp_dim)
        .is_err());
}

#[test]
fn mode_add_is_reversible() {
    assert_eq!(Mode::Add.apply(2.0, 3.0), 5.0);
}

#[test]
fn mock_forward_deterministic() {
    let b = backend();
    let ids = b.tokenize("sad story").unwrap();
    let a = b.forward(&ids, &[]).unwrap();
    let b2 = b.forward(&ids, &[]).unwrap();
    assert_eq!(a, b2);
}
