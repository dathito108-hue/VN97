# P2 Real Pilot Result / P3 Entry

P2 completed successfully on GitHub Actions run `35987320466`.

The exact sealed corpus was:

- VN97CORPUS1: `dd32ff3c62c9f0f0b547f25a57ab4d6363a1bf732da1387db4e7553399bbaf9b`
- training records: 1,818
- validation records: 92
- release records: 90

The exact P2 candidate was:

- candidate ID: `93aa23b1dc87b547`
- d_model: 192
- layers: 6
- d_state: 16
- factorized embedding rank: 96
- learning rate: 3e-4
- seed: 97

Measured P2 result:

- parameters: 1,379,520
- training steps: 2,010
- training target tokens: 373,748
- training mean loss: 7.2403230579
- training final loss: 7.1219000816
- validation mean loss: 6.6261675947
- validation top-1: 0.0722794033
- validation target tokens: 8,647
- release mean loss: 6.6490118611
- release top-1: 0.0594106920
- release target tokens: 10,419
- VN97MI1 bytes: 2,138,367
- recurrent state bytes: 73,728

Identity:

- VN97CK1: `1edf7d43c019b303676babb7722693f278f6a102a54ec80848664e4f93761675`
- VN97TK1: `ce0d6fb36f910a7b4bbf7221731c6c5bee8d18a0774e9432dded82070fe0d460`
- VN97MI1: `8b255514710cf558281ebda8fecc8f8c644e7b7456052755e673fe7b501fd6b4`
- VN97P2RUN1: `e0509769f3f819516c7e41c28769053673f634e41051d8e469b95bd8421db5a2`

P2 is therefore closed as a real pipeline/learning proof. These metrics are not a
claim that the pilot is production-quality intelligence.

## P3 entry

P3 expands the same already-reviewed pinned sources to their bounded full-data caps
before spending rented GPU compute:

- Vietnamese dialogue: up to 2,009 accepted records;
- Databricks Dolly: up to 15,011 accepted records;
- GSM8K train: up to 7,473 accepted records.

The same deterministic canonicalization, identity-contamination filter, split hashing,
deduplication and VN97CORPUS1 leakage gate remain in force.

P3 candidate geometry is frozen in:

`configs/p3-language-production.vn97campdef1.json`

Four same-architecture candidates are compared. No alternate backend or manual winner
override is allowed.

The P3 campaign should use a larger tokenizer/context than P2, with a bounded GPU
campaign. Final quality is not decided by P3 loss alone; the selected winner must
still pass P4 assistant-task evaluation before release packaging.
