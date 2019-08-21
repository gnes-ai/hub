from gnes.preprocessor.text.base import BaseTextPreprocessor


class MyPreprocessor2(BaseTextPreprocessor):

    def __init__(self, bar, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bar = bar

    def apply(self, doc: 'gnes_pb2.Document') -> None:
        super().apply(doc)
        doc.raw_text += self.bar
        self.logger.info(doc.raw_text)
