import gc
import logging
from typing import Union, Optional, Tuple

from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from transformers import (
    DebertaV2ForSequenceClassification,
    DebertaV2PreTrainedModel,
    DebertaV2Model,
    AutoConfig,
    DebertaV2Tokenizer,
    BertModel,
    BertPreTrainedModel,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.deberta_v2.modeling_deberta_v2 import (
    ContextPooler,
    DEBERTA_INPUTS_DOCSTRING,
    _CHECKPOINT_FOR_DOC,
    _CONFIG_FOR_DOC,
)
from torch import nn
import torch
from transformers.utils import (
    add_start_docstrings_to_model_forward,
    add_code_sample_docstrings,
)

from dga_classifier import get_settings
from dga_classifier.data.dataset import label_to_int
from dga_classifier.models.DeBERTa.sift import hook_sift_layer, AdversarialLearner

settings = get_settings()

logger = logging.getLogger("dga_classifier.DeBERTa.model")


# Copied from transformers.models.deberta.modeling_deberta.DropoutContext
class DropoutContext(object):
    def __init__(self):
        self.dropout = 0
        self.mask = None
        self.scale = 1
        self.reuse_mask = True


# Copied from transformers.models.deberta.modeling_deberta.get_mask
def get_mask(input, local_context):
    if not isinstance(local_context, DropoutContext):
        dropout = local_context
        mask = None
    else:
        dropout = local_context.dropout
        dropout *= local_context.scale
        mask = local_context.mask if local_context.reuse_mask else None

    if dropout > 0 and mask is None:
        mask = (1 - torch.empty_like(input).bernoulli_(1 - dropout)).to(torch.bool)

    if isinstance(local_context, DropoutContext):
        if local_context.mask is None:
            local_context.mask = mask

    return mask, dropout


# Copied from transformers.models.deberta.modeling_deberta.XDropout
class XDropout(torch.autograd.Function):
    """Optimized dropout function to save computation and memory by using mask operation instead of multiplication."""

    @staticmethod
    def forward(ctx, input, local_ctx):
        mask, dropout = get_mask(input, local_ctx)
        ctx.scale = 1.0 / (1 - dropout)
        if dropout > 0:
            ctx.save_for_backward(mask)
            return input.masked_fill(mask, 0) * ctx.scale
        else:
            return input

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.scale > 1:
            (mask,) = ctx.saved_tensors
            return grad_output.masked_fill(mask, 0) * ctx.scale, None
        else:
            return grad_output, None

    @staticmethod
    def symbolic(
        g: torch._C.Graph,
        input: torch._C.Value,
        local_ctx: Union[float, DropoutContext],
    ) -> torch._C.Value:
        from torch.onnx import symbolic_opset12

        dropout_p = local_ctx
        if isinstance(local_ctx, DropoutContext):
            dropout_p = local_ctx.dropout
        # StableDropout only calls this function when training.
        train = True
        # TODO: We should check if the opset_version being used to export
        # is > 12 here, but there's no good way to do that. As-is, if the
        # opset_version < 12, export will fail with a CheckerError.
        # Once https://github.com/pytorch/pytorch/issues/78391 is fixed, do something like:
        # if opset_version < 12:
        #   return torch.onnx.symbolic_opset9.dropout(g, input, dropout_p, train)
        return symbolic_opset12.dropout(g, input, dropout_p, train)


# Copied from transformers.models.deberta.modeling_deberta.StableDropout
class StableDropout(nn.Module):
    """
    Optimized dropout module for stabilizing the training

    Args:
        drop_prob (float): the dropout probabilities
    """

    def __init__(self, drop_prob):
        super().__init__()
        self.drop_prob = drop_prob
        self.count = 0
        self.context_stack = None

    def forward(self, x):
        """
        Call the module

        Args:
            x (`torch.tensor`): The input tensor to apply dropout
        """
        if self.training and self.drop_prob > 0:
            return XDropout.apply(x, self.get_context())
        return x

    def clear_context(self):
        self.count = 0
        self.context_stack = None

    def init_context(self, reuse_mask=True, scale=1):
        if self.context_stack is None:
            self.context_stack = []
        self.count = 0
        for c in self.context_stack:
            c.reuse_mask = reuse_mask
            c.scale = scale

    def get_context(self):
        if self.context_stack is not None:
            if self.count >= len(self.context_stack):
                self.context_stack.append(DropoutContext())
            ctx = self.context_stack[self.count]
            ctx.dropout = self.drop_prob
            self.count += 1
            return ctx
        else:
            return self.drop_prob


class DeBERTaForDGAClassificationSiFT(DebertaV2PreTrainedModel):
    def __init__(self):
        config = AutoConfig.from_pretrained("microsoft/deberta-v3-base")
        super().__init__(config)
        self.model = DeBERTaForDGAClassification()
        self.adv_modules = hook_sift_layer(self.model, hidden_size=768)
        self.adv = AdversarialLearner(self.model, self.adv_modules)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.model(
            input_ids=input_ids,
            labels=labels,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Create DeBERTa model
        def logits_fn(model, *wargs, **kwargs):
            output = model(*wargs, **kwargs)
            return output.logits

        logits = outputs.logits
        loss = outputs.loss

        if self.training:
            loss = loss + self.adv.loss(
                logits,
                logits_fn,
                input_ids=input_ids,
                labels=labels,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class DeBERTaForDGAClassification(DebertaV2PreTrainedModel):
    def __init__(self, config=None):
        if config is None:
            config = AutoConfig.from_pretrained("microsoft/deberta-v3-base")
        super().__init__(config)

        # num_labels = getattr(config, "num_labels", 2)
        self.num_labels = 2

        # self.deberta = DebertaV2Model.from_pretrained(
        #     "microsoft/deberta-v3-base",
        # )
        self.deberta = DebertaV2Model(config)
        self.pooler = ContextPooler(config)
        output_dim = self.pooler.output_dim

        # self.norm_classifier = nn.BatchNorm1d(output_dim)

        # self.dropout_classifier = nn.Dropout(0.1)
        self.classifier = nn.Linear(output_dim, self.num_labels)
        self.sigmoid = nn.Sigmoid()
        self.relu_classifier = nn.ReLU()

        # self.relu = nn.ReLU(inplace=True)
        # self.maxpool_p = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        # self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        #
        # self.layer0 = Residual(output_dim, 256, kernel_size=4)
        # self.layer1 = Residual(256, 256, kernel_size=3)
        # self.layer2 = Residual(256, 256)
        # self.layer3 = Residual(256, 256)
        # self.layer4 = Residual(256, 256)
        #
        # self.ll = nn.Linear(512, 1)

        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out
        self.dropout = StableDropout(drop_out)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.deberta.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        self.deberta.set_input_embeddings(new_embeddings)

    # Copied from transformers.models.deberta.modeling_deberta.DebertaForSequenceClassification.forward with Deberta->DebertaV2
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.deberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        encoder_layer = outputs[0]
        pooled_output = self.pooler(encoder_layer)
        pooled_output = self.dropout(pooled_output)

        # x = self.norm_classifier(pooled_output)
        # x = self.relu_classifier(x)
        # x = self.dropout_classifier(x)
        logits = self.classifier(pooled_output)

        # x = outputs[:][0]
        # x = x.permute(0, 2, 1)
        #
        # x = self.layer0(x)
        # x = self.maxpool_p(x)
        #
        # x = self.layer1(x)
        # x = self.maxpool(x)
        #
        # x = self.layer2(x)
        # x = self.maxpool(x)
        # x = self.layer3(x)
        # x = self.maxpool(x)
        # x = self.layer4(x)
        # x = self.maxpool(x)
        #
        # x = torch.flatten(x, 1, 2)
        #
        # logits = self.ll(x).view(-1)

        # return x.flatten()
        prob = self.sigmoid(logits)

        # loss_fn = nn.BCELoss()
        # loss = loss_fn(prob.view(logits.shape[0], -1), labels.float())
        # loss = loss_fn(prob, labels.float())

        loss = None

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class MResNet(nn.Module):
    def __init__(
        self, embedding_dim: int = 128, vocab_size: int = 38, out_channels: int = 256
    ):
        super().__init__()

        self.name = "MResNet"
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.out_channels = out_channels
        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size, embedding_dim=self.embedding_dim
        )

        self.relu = nn.ReLU(inplace=True)
        self.maxpool_p = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.layer0 = Residual(128, 256, kernel_size=4)
        self.layer1 = Residual(256, 256, kernel_size=3)
        self.layer2 = Residual(256, 256)
        self.layer3 = Residual(256, 256)
        self.layer4 = Residual(256, 256)

        self.ll = nn.Linear(512, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x.long())
        x = x.permute(0, 2, 1)

        x = self.layer0(x)
        x = self.maxpool_p(x)

        x = self.layer1(x)
        x = self.maxpool(x)

        x = self.layer2(x)
        x = self.maxpool(x)
        x = self.layer3(x)
        x = self.maxpool(x)
        x = self.layer4(x)
        x = self.maxpool(x)

        x = torch.flatten(x, 1, 2)

        x = self.ll(x)
        return x.flatten()


class Residual(nn.Module):
    def __init__(self, input_dim: int = 256, out_channels: int = 256, kernel_size=2):
        super().__init__()
        self.input_dim = input_dim
        self.out_channels = out_channels

        self.channel_adjust = nn.Conv1d(
            in_channels=input_dim,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding="same",
        )

        self.conv1 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding="same",
        )
        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding="same",
        )

        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, batch_x: torch.Tensor) -> torch.Tensor:
        residual = batch_x

        if self.out_channels != self.input_dim:
            residual = self.channel_adjust(batch_x)

        x = self.conv1(residual)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x = residual + x

        x = self.relu(x)

        return x


class DeBERTaCNNForDGAClassification(DebertaV2PreTrainedModel):
    def __init__(self):
        pretrained_model_name_or_path = "microsoft/deberta-v3-base"
        if settings.use_pretrained_bert_model:
            pretrained_model_name_or_path = settings.pretrained_model_path

        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)

        super().__init__(config)

        # num_labels = getattr(config, "num_labels", 2)

        num_labels = 2
        if settings.classification_task == "multiclass":
            num_labels = len(label_to_int)

        self.num_labels = num_labels
        self.deberta = DebertaV2Model.from_pretrained(pretrained_model_name_or_path)

        self.pooler = ContextPooler(config)
        output_dim = self.pooler.output_dim
        filter_sizes = [1, 2, 3, 4, 5]
        num_filters = 32
        self.convs1 = nn.ModuleList(
            [nn.Conv2d(4, num_filters, (K, output_dim)) for K in filter_sizes]
        )
        self.norm_classifier = nn.BatchNorm1d(output_dim)
        self.dropout_classifier = nn.Dropout(0.1)
        self.classifier = nn.Linear(len(filter_sizes) * num_filters, num_labels)
        # self.classifier = nn.Linear(output_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu_classifier = nn.ReLU()

        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out
        self.dropout = StableDropout(drop_out)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.deberta.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        self.deberta.set_input_embeddings(new_embeddings)

    # Copied from transformers.models.deberta.modeling_deberta.DebertaForSequenceClassification.forward with Deberta->DebertaV2
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.deberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )
        x = outputs.hidden_states[-4:]
        x = torch.stack(x, dim=1)
        x = [nn.functional.relu(conv(x)).squeeze(3) for conv in self.convs1]
        x = [nn.functional.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout_classifier(x)
        logits = self.classifier(x)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
        )


class DeBERTaCNN12ForDGAClassification(DebertaV2PreTrainedModel):
    def __init__(self, config=None):
        if config is None:
            config = AutoConfig.from_pretrained("microsoft/deberta-v3-base")
        super().__init__(config)

        # num_labels = getattr(config, "num_labels", 2)
        # self.num_labels = num_labels

        # self.deberta = DebertaV2Model.from_pretrained(
        #     "microsoft/deberta-v3-base",
        # )
        self.deberta = DebertaV2Model(config)
        self.pooler = ContextPooler(config)
        output_dim = self.pooler.output_dim
        self.norm_classifier = nn.BatchNorm1d(output_dim)

        self.conv = nn.Conv2d(
            in_channels=13, out_channels=13, kernel_size=(3, 768), padding=1
        )
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1)
        self.fc = nn.Linear(793, 2)
        self.flat = nn.Flatten()
        self.dropout = nn.Dropout(0.1)
        # self.relu = nn.ReLU(inplace=True)
        # self.maxpool_p = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        # self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        #
        # self.layer0 = Residual(output_dim, 256, kernel_size=4)
        # self.layer1 = Residual(256, 256, kernel_size=3)
        # self.layer2 = Residual(256, 256)
        # self.layer3 = Residual(256, 256)
        # self.layer4 = Residual(256, 256)
        #
        # self.ll = nn.Linear(512, 1)

        # drop_out = getattr(config, "cls_dropout", None)
        # drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out
        # self.dropout = StableDropout(drop_out)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.deberta.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        self.deberta.set_input_embeddings(new_embeddings)

    # Copied from transformers.models.deberta.modeling_deberta.DebertaForSequenceClassification.forward with Deberta->DebertaV2
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.deberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        all_layers = outputs.hidden_states

        x = torch.transpose(
            torch.cat(tuple([t.unsqueeze(0) for t in all_layers]), 0), 0, 1
        )
        x = self.pool(self.dropout(self.relu(self.conv(self.dropout(x)))))
        logits = self.fc(self.dropout(self.flat(self.dropout(x))))
        # x = outputs[:][0]
        # x = x.permute(0, 2, 1)
        #
        # x = self.layer0(x)
        # x = self.maxpool_p(x)
        #
        # x = self.layer1(x)
        # x = self.maxpool(x)
        #
        # x = self.layer2(x)
        # x = self.maxpool(x)
        # x = self.layer3(x)
        # x = self.maxpool(x)
        # x = self.layer4(x)
        # x = self.maxpool(x)
        #
        # x = torch.flatten(x, 1, 2)
        #
        # logits = self.ll(x).view(-1)

        # return x.flatten()

        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, 2), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class DeBERTaBiLSTMForDGAClassification(DebertaV2PreTrainedModel):
    def __init__(self, config=None):
        if config is None:
            config = AutoConfig.from_pretrained("microsoft/deberta-v3-base")

            # config = AutoConfig.from_pretrained("bert-base-uncased")
        super().__init__(config)

        # num_labels = getattr(config, "num_labels", 2)
        # self.num_labels = num_labels

        # self.deberta = DebertaV2Model.from_pretrained(
        #     "microsoft/deberta-v3-base",
        # )
        self.deberta = DebertaV2Model.from_pretrained("microsoft/deberta-v3-base")
        # self.bert = BertModel(config)
        # self.pooler = ContextPooler(config)
        # output_dim = self.pooler.output_dim

        # self.norm_classifier = nn.BatchNorm1d(output_dim)
        self.dropout_classifier = nn.Dropout(0.1)
        # self.classifier = nn.Linear(output_dim, 2)
        hidden_size = 128
        num_layers = 2
        bidirectional = True
        num_classes = 2
        self.lstm = nn.LSTM(
            input_size=768,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(
            hidden_size * 2 if bidirectional else hidden_size, num_classes
        )
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.deberta.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        self.deberta.set_input_embeddings(new_embeddings)

    # Copied from transformers.models.deberta.modeling_deberta.DebertaForSequenceClassification.forward with Deberta->DebertaV2
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.deberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        encoder_layer = outputs.last_hidden_state

        lstm_out, x = self.lstm(encoder_layer)

        batch_size = input_ids.shape[0]

        sequence_lengths = (torch.eq(input_ids, 0).long().argmax(-1) - 1).to(
            encoder_layer.device
        )
        logits = lstm_out[
            torch.arange(batch_size, device=encoder_layer.device), sequence_lengths
        ]

        # pooled_output = self.pooler(encoder_layer)
        # pooled_output = self.dropout(pooled_output)
        # output = self.dropout(pooled_output)

        logits = self.classifier(logits)
        # logits = self.classifier(output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class DeBERTaResNetForDGAClassification(DebertaV2PreTrainedModel):
    def __init__(self, config=None):
        pretrained_model_name_or_path = "microsoft/deberta-v3-base"
        if settings.use_pretrained_bert_model:
            pretrained_model_name_or_path = settings.pretrained_model_path

        if config is None:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path)

        super().__init__(config)

        # num_labels = getattr(config, "num_labels", 2)

        num_labels = 2
        if settings.classification_task == "multiclass":
            num_labels = len(label_to_int)

        self.num_labels = num_labels
        self.deberta = DebertaV2Model.from_pretrained(pretrained_model_name_or_path)

        # self.deberta = DebertaV2Model.from_pretrained(
        #     "microsoft/deberta-v3-base",
        # )
        self.pooler = ContextPooler(config)
        output_dim = self.pooler.output_dim

        # self.norm_classifier = nn.BatchNorm1d(output_dim)

        # self.dropout_classifier = nn.Dropout(0.1)
        self.classifier = nn.Linear(output_dim, self.num_labels)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool_p = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.layer0 = Residual(output_dim, 256, kernel_size=4)
        self.layer1 = Residual(256, 256, kernel_size=3)
        self.layer2 = Residual(256, 256)
        self.layer3 = Residual(256, 256)
        self.layer4 = Residual(256, 256)

        self.ll = nn.Linear(512, 2)

        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out
        self.dropout = StableDropout(drop_out)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.deberta.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        self.deberta.set_input_embeddings(new_embeddings)

    # Copied from transformers.models.deberta.modeling_deberta.DebertaForSequenceClassification.forward with Deberta->DebertaV2
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.deberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        encoder_layer = outputs[0]
        pooled_output = self.pooler(encoder_layer)
        pooled_output = self.dropout(pooled_output)

        # x = self.norm_classifier(pooled_output)
        # x = self.relu_classifier(x)
        # x = self.dropout_classifier(x)
        logits = self.classifier(pooled_output)

        x = outputs[:][0]
        x = x.permute(0, 2, 1)

        x = self.layer0(x)
        x = self.maxpool_p(x)

        x = self.layer1(x)
        x = self.maxpool(x)

        x = self.layer2(x)
        x = self.maxpool(x)
        x = self.layer3(x)
        x = self.maxpool(x)
        x = self.layer4(x)
        x = self.maxpool(x)

        x = torch.flatten(x, 1, 2)

        logits = self.ll(x)

        # return x.flatten()

        # loss_fn = nn.BCELoss()
        # loss = loss_fn(prob.view(logits.shape[0], -1), labels.float())
        # loss = loss_fn(prob, labels.float())

        loss = None

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

def get_tokenizer():
    logger.info(f"Get custom tokenizer from {settings.deberta_vocab_path}")
    tokenizer = DebertaV2Tokenizer(vocab_file=settings.deberta_vocab_path)
    return tokenizer
